# import debugpy; debugpy.connect(('127.0.0.1', 8885))

"""
EDiVal Instruction Following Reward Server

A Flask-based service deployment that calls the independent GroundingDINO and vLLM services.

Dependent services:
    1. GroundingDINO Server: http://localhost:12343
    2. vLLM Server: http://localhost:8000 (TP=8)

Before starting this service, please start:
    bash start_servers.sh        # starts vLLM
    bash start_groundingdino.sh  # starts GroundingDINO

Start this service:
    python edival_reward_server.py

Service endpoints:
    POST /mode/instruction_following
    Request body (pickle-serialized):
    {
        "ref_images": [bytes, bytes, ...],  # Reference images (JPEG bytes)
        "images": [bytes, bytes, ...],      # Generated images (JPEG bytes)
        "prompts": [str, str, ...],         # Natural language instructions
        "metadatas": [                       # Metadata, must include
            {
                "task_type": str,           # e.g. "subject_add"
                "formatted_instruction": str # e.g. "Add [dog] on the [left] of [tree]"
            },
            ...
        ]
    }
    Response body (pickle-serialized):
    {
        "scores": [float, float, ...],      # Score of 0 or 1
        "reasons": [str, str, ...]          # Evaluation reasons
    }
"""

import os
import sys
import pickle
import base64
from io import BytesIO
from typing import List, Dict, Tuple, Optional
from PIL import Image
from flask import Flask, request
import traceback
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.nn.functional as F
import numpy as np

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Read the model path from the environment variable, matching start_servers.sh
try:
    VLLM_MODEL_PATH = os.environ["VLLM_MODEL_PATH"]
except:
    print('Please make sure the VLLM_MODEL_PATH environment variable is exported~')

# DINOv3 imports
try:
    from transformers import AutoImageProcessor, AutoModel
except Exception:
    AutoImageProcessor = None
    AutoModel = None

app = Flask(__name__)

# Configuration
GROUNDINGDINO_URL = os.getenv("GROUNDINGDINO_URL", "http://localhost:12343")
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1")
BATCH_SIZE = int(os.getenv("EDIVAL_BATCH_SIZE", "8"))  # Parallel processing batch size
VLM_TEMPERATURE = float(os.getenv("VLM_TEMPERATURE", "0.0"))  # VLM sampling temperature, 0.0 = greedy decoding

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=BATCH_SIZE)

# ========== DINOv3 Model Loading ==========

def load_dinov3_model(device=None):
    """Load the DINOv3 model used for consistency evaluation."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if AutoImageProcessor is None or AutoModel is None:
        print("[Warning] transformers not available, DINOv3 consistency will be disabled")
        return None

    dinov3_path = os.environ.get("DINOV3_PATH")
    if not dinov3_path:
        print("[Warning] DINOV3_PATH environment variable not set, DINOv3 consistency will be disabled")
        return None

    try:
        dino_processor = AutoImageProcessor.from_pretrained(dinov3_path)
        dino_model = AutoModel.from_pretrained(dinov3_path)
        dino_model.eval().to(device)
        print(f"[DINOv3] Model loaded on {device}")
        return (dino_model, dino_processor)
    except Exception as e:
        print(f"[Warning] Failed to load DINOv3 model: {e}")
        return None

# Load the DINOv3 model globally
DINOV3_BUNDLE = load_dinov3_model()


def _l2norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize a tensor along the last dimension."""
    return F.normalize(x, p=2, dim=-1, eps=eps)


def _masked_global_cosine(tokens1: torch.Tensor, w1: torch.Tensor,
                          tokens2: torch.Tensor, w2: torch.Tensor,
                          weight_thresh: float = 0.0) -> float:
    """Compute cosine similarity between two sets of tokens using soft per-token weights."""
    if weight_thresh > 0:
        m1 = (w1 >= weight_thresh).float()
        m2 = (w2 >= weight_thresh).float()
        w1, w2 = w1 * m1, w2 * m2

    # Normalize token features before pooling
    t1 = _l2norm(tokens1.squeeze(0))  # (N, D)
    t2 = _l2norm(tokens2.squeeze(0))  # (N, D)

    # Weighted average (masked GAP)
    w1e = w1.squeeze(0).unsqueeze(-1)  # (N,1)
    w2e = w2.squeeze(0).unsqueeze(-1)
    f1 = (t1 * w1e).sum(dim=0) / (w1e.sum() + 1e-6)  # (D,)
    f2 = (t2 * w2e).sum(dim=0) / (w2e.sum() + 1e-6)  # (D,)

    sim = F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
    return float(sim)


def _calculate_dinov3_similarity(dino_bundle, object1: Image.Image, object2: Image.Image, alpha: float = 0.5):
    """Compute similarity with DINOv3 features using HF AutoModel/Processor."""
    if not isinstance(object1, Image.Image) or not isinstance(object2, Image.Image):
        raise ValueError("object1 and object2 should be PIL Images")

    dino_model, dino_processor = dino_bundle
    device = next(dino_model.parameters()).device

    with torch.no_grad():
        inputs = dino_processor(images=[object1, object2], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        outputs = dino_model(pixel_values=pixel_values)
        feats = outputs.last_hidden_state
        if feats is None:
            feats = outputs[0]

        cls1 = feats[0, 0:1, :]  # [1, D]
        cls2 = feats[1, 0:1, :]
        semantic_sim = F.cosine_similarity(cls1, cls2, dim=1)  # [1]

        patch1 = feats[0, 1:, :].mean(dim=0, keepdim=True)  # [1, D]
        patch2 = feats[1, 1:, :].mean(dim=0, keepdim=True)
        texture_sim = F.cosine_similarity(patch1, patch2, dim=1)  # [1]

        combined_similarity = alpha * semantic_sim + (1 - alpha) * texture_sim  # [1]
        return combined_similarity.item()


def _crop_object_from_image(image: Image.Image, box: List[float]) -> Image.Image:
    """Crop object from image using normalized box coordinates."""
    if not isinstance(image, Image.Image):
        raise ValueError("image should be a PIL Image")

    width, height = image.size
    x1 = int(box[0] * width)
    y1 = int(box[1] * height)
    x2 = int(box[2] * width)
    y2 = int(box[3] * height)

    # Ensure coordinates are within bounds
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    return image.crop((x1, y1, x2, y2))


def _get_background_mask(image: Image.Image, boxes: List[List[float]]) -> Image.Image:
    """Create a mask image (L mode) where 255=keep background, 0=mask objects."""
    from PIL import ImageDraw

    mask = Image.new('L', image.size, 255)  # White mask (keep all)
    if not boxes:
        return mask

    mask_draw = ImageDraw.Draw(mask)
    width, height = image.size

    for box in boxes:
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)

        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        mask_draw.rectangle([x1, y1, x2, y2], fill=0)  # Black = remove

    return mask


def _calculate_background_dinov3(dino_bundle, src_img: Image.Image, target_img: Image.Image,
                                  mask_img: Image.Image) -> Dict:
    """Compute DINOv3 masked background similarity."""
    import torchvision.transforms as transforms
    import math

    if dino_bundle is None:
        return {'bg_dinov3_masked_similarity': None, 'bg_dinov3_masked_patches': 0}

    dino_model, dino_processor = dino_bundle
    device = next(dino_model.parameters()).device

    try:
        with torch.no_grad():
            inputs = dino_processor(images=[src_img, target_img], return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            _, _, Hpixels, Wpixels = pixel_values.shape

            outputs = dino_model(pixel_values=pixel_values)
            feats = outputs.last_hidden_state
            if feats is None:
                feats = outputs[0]

            # Determine patch grid size
            Np = feats.shape[1] - 1
            aspect = float(Hpixels) / float(max(1, Wpixels))
            best_diff = None
            Hp, Wp = 1, Np
            for w in range(1, int(math.sqrt(Np)) + 1):
                if Np % w != 0:
                    continue
                h = Np // w
                for hh, ww in ((h, w), (w, h)):
                    diff = abs((hh / max(1.0, ww)) - aspect)
                    if best_diff is None or diff < best_diff:
                        best_diff = diff
                        Hp, Wp = int(hh), int(ww)

            # Resize mask to model input resolution
            mask_to_model = mask_img.resize((Wpixels, Hpixels), resample=Image.NEAREST)
            mask_tensor_img = transforms.ToTensor()(mask_to_model).unsqueeze(0).to(device)

            # Pool to patch grid
            kH = Hpixels // Hp
            kW = Wpixels // Wp
            if Hpixels % Hp == 0 and Wpixels % Wp == 0 and kH > 0 and kW > 0:
                weights = F.avg_pool2d(mask_tensor_img, kernel_size=(kH, kW), stride=(kH, kW))
            else:
                weights = F.interpolate(mask_tensor_img, size=(Hp, Wp), mode='area')
            weights = weights.squeeze(0).squeeze(0).clamp(0.0, 1.0)

            # Reshape tokens and compute similarity
            tokens_src = feats[0, 1:, :].reshape(Hp, Wp, -1)
            tokens_tgt = feats[1, 1:, :].reshape(Hp, Wp, -1)
            ts_flat = tokens_src.reshape(-1, tokens_src.shape[-1]).unsqueeze(0)
            tt_flat = tokens_tgt.reshape(-1, tokens_tgt.shape[-1]).unsqueeze(0)
            w_flat = weights.reshape(1, -1)

            sim = _masked_global_cosine(ts_flat, w_flat, tt_flat, w_flat, weight_thresh=0.5)
            n_patches = int((w_flat > 0).sum().item())

            return {'bg_dinov3_masked_similarity': sim, 'bg_dinov3_masked_patches': n_patches}
    except Exception as e:
        print(f"[Warning] DINOv3 background similarity failed: {e}")
        return {'bg_dinov3_masked_similarity': None, 'bg_dinov3_masked_patches': 0}


def _calculate_object_consistency(dino_bundle, grounding_detections_src, grounding_detections_tgt,
                                   src_img: Image.Image, target_img: Image.Image) -> Dict:
    """Calculate object consistency using DINOv3 similarity and L1 loss."""
    import torchvision.transforms as transforms

    dinov3_similarity = []
    l1_consistency = []

    # Use boxes from src image
    for box in grounding_detections_src.get("box", []):
        src_object = _crop_object_from_image(src_img, box)
        target_object = _crop_object_from_image(target_img, box)

        # DINOv3 similarity
        if dino_bundle is not None:
            try:
                similarity = _calculate_dinov3_similarity(dino_bundle, src_object, target_object)
                dinov3_similarity.append(similarity)
            except Exception as e:
                print(f"[Warning] DINOv3 similarity failed: {e}")

        # L1 consistency
        try:
            transform = transforms.ToTensor()
            src_tensor = transform(src_object)
            target_tensor = transform(target_object)
            l1_loss = F.l1_loss(src_tensor, target_tensor)
            l1_consistency.append(1 - l1_loss.item())
        except Exception as e:
            print(f"[Warning] L1 consistency failed: {e}")

    return {
        "object_dinov3_consistency": dinov3_similarity,
        "object_dinov3_consistency_mean": float(np.mean(dinov3_similarity)) if dinov3_similarity else None,
        "object_l1_consistency": l1_consistency,
        "object_l1_consistency_mean": float(np.mean(l1_consistency)) if l1_consistency else None
    }


def _calculate_background_consistency(grounding_detections_src, grounding_detections_tgt,
                                       src_img: Image.Image, target_img: Image.Image,
                                       dino_bundle=None) -> Dict:
    """Calculate background consistency using L1 and DINOv3."""
    import torchvision.transforms as transforms

    # Union all boxes from both images
    all_boxes = grounding_detections_src.get("box", []) + grounding_detections_tgt.get("box", [])

    if not all_boxes:
        return {
            "bg_l1_consistency": None,
            "bg_dinov3_masked_similarity": None,
            "bg_dinov3_masked_patches": 0,
            "total_boxes_used": 0
        }

    # Create mask
    mask_img = _get_background_mask(src_img, all_boxes)

    # Convert to tensors
    transform = transforms.ToTensor()
    mask_tensor = transform(mask_img).expand(3, -1, -1)
    valid_count = mask_tensor.sum()

    if valid_count.item() == 0:
        return {
            "bg_l1_consistency": None,
            "bg_dinov3_masked_similarity": None,
            "bg_dinov3_masked_patches": 0,
            "total_boxes_used": len(all_boxes)
        }

    # Masked L1
    src_tensor = transform(src_img)
    target_tensor = transform(target_img)
    diff = (src_tensor - target_tensor).abs()
    l1_loss = (diff * mask_tensor).sum() / valid_count
    bg_consistency = 1 - l1_loss.item()

    # DINOv3 background similarity
    dino_result = _calculate_background_dinov3(dino_bundle, src_img, target_img, mask_img)

    return {
        "bg_l1_consistency": bg_consistency,
        "bg_dinov3_masked_similarity": dino_result['bg_dinov3_masked_similarity'],
        "bg_dinov3_masked_patches": dino_result['bg_dinov3_masked_patches'],
        "total_boxes_used": len(all_boxes)
    }


def evaluate_consistency_single(src_img: Image.Image, target_img: Image.Image,
                                 unchanged_objects: List[str], all_objects: List[str]) -> Tuple[Dict, Dict]:
    """
    Evaluate consistency for a single pair of images.
    Optimization: uses call_groundingdino_multi for batched detection, requiring only
    3 HTTP calls (instead of the previous N + 2*M sequential calls).

    Returns:
        (object_result, background_result): dicts containing the consistency metrics
    """
    if src_img.size != target_img.size:
        target_img = target_img.resize(src_img.size, Image.LANCZOS)

    if isinstance(unchanged_objects, str):
        unchanged_objects = [obj.strip() for obj in unchanged_objects.split(".") if obj.strip()]
    if isinstance(all_objects, str):
        all_objects = [obj.strip() for obj in all_objects.split(".") if obj.strip()]

    if not unchanged_objects or not all_objects:
        return (
            {
                "object_dinov3_consistency_mean": None,
                "object_l1_consistency_mean": None
            },
            {
                "bg_l1_consistency": None,
                "bg_dinov3_masked_similarity": None
            }
        )

    # 3 batched GroundingDINO calls (replacing the original N + 2*M sequential calls)
    src_detections = call_groundingdino_multi(src_img, unchanged_objects, threshold=0.35)
    all_detections_src = call_groundingdino_multi(src_img, all_objects, threshold=0.35)
    all_detections_tgt = call_groundingdino_multi(target_img, all_objects, threshold=0.35)

    object_result = _calculate_object_consistency(
        DINOV3_BUNDLE, src_detections, all_detections_tgt,
        src_img, target_img
    )

    background_result = _calculate_background_consistency(
        all_detections_src, all_detections_tgt,
        src_img, target_img,
        dino_bundle=DINOV3_BUNDLE
    )

    return object_result, background_result


def pil_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to a base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def call_groundingdino(image: Image.Image, target_object: str, threshold: float = 0.3,
                       return_all: bool = False, delete_large_box: bool = False) -> Dict:
    """
    Call the GroundingDINO service to perform object detection

    Args:
        image: PIL Image
        target_object: name of the target object
        threshold: detection threshold
        return_all: whether to return all detection results
        delete_large_box: whether to drop overly large boxes

    Returns:
        dict of detection results
    """
    try:
        image_b64 = pil_to_base64(image)
        
        response = requests.post(
            f"{GROUNDINGDINO_URL}/detect",
            json={
                "image": image_b64,
                "target_object": target_object,
                "threshold": threshold,
                "return_all": return_all,
                "delete_large_box": delete_large_box
            },
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        if result.get("success"):
            return result.get("detections", {"label": [], "score": [], "box": [], "center": []})
        else:
            print(f"[GroundingDINO Error] {result.get('error', 'Unknown error')}")
            return {"label": [], "score": [], "box": [], "center": []}
            
    except Exception as e:
        print(f"[GroundingDINO Call Error] {e}")
        return {"label": [], "score": [], "box": [], "center": []}


def call_groundingdino_multi(image: Image.Image, objects: List[str], threshold: float = 0.35) -> Dict:
    """
    Call the GroundingDINO service once to detect multiple objects (mirrors the
    reference implementation's _detect_multiple_objects_from_img). All object
    names are joined with " . " into a single prompt, requiring only one HTTP call.

    Args:
        image: PIL Image
        objects: list of object names, e.g. ["dog", "cat", "tree"]
        threshold: detection threshold

    Returns:
        dict of detection results {"label": [...], "score": [...], "box": [...], "center": [...]}
    """
    if not objects:
        return {"label": [], "score": [], "box": [], "center": []}

    try:
        image_b64 = pil_to_base64(image)

        response = requests.post(
            f"{GROUNDINGDINO_URL}/detect",
            json={
                "image": image_b64,
                "target_objects": objects,
                "threshold": threshold,
                "return_all": True,
            },
            timeout=30
        )
        response.raise_for_status()

        result = response.json()
        if result.get("success"):
            return result.get("detections", {"label": [], "score": [], "box": [], "center": []})
        else:
            print(f"[GroundingDINO Multi Error] {result.get('error', 'Unknown error')}")
            return {"label": [], "score": [], "box": [], "center": []}

    except Exception as e:
        print(f"[GroundingDINO Multi Call Error] {e}")
        return {"label": [], "score": [], "box": [], "center": []}


def call_vlm(image: Image.Image, prompt: str) -> str:
    """
    Call the vLLM service for visual question answering

    Args:
        image: PIL Image
        prompt: text prompt

    Returns:
        the model's answer (lowercased)
    """
    try:
        image_b64 = pil_to_base64(image)
        
        # Use the OpenAI API format
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": VLLM_MODEL_PATH,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_b64}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            "temperature": VLM_TEMPERATURE,
            "max_tokens": 1024
        }

        response = requests.post(
            f"{VLLM_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            answer = result["choices"][0]["message"]["content"].strip().lower()
            return answer
        else:
            return "no"

    except Exception as e:
        print(f"[VLM Call Error] {e}")
        return "no"


def call_vlm_2image(image1: Image.Image, image2: Image.Image, prompt: str) -> str:
    """
    Call the vLLM service for two-image comparison
    """
    try:
        image1_b64 = pil_to_base64(image1)
        image2_b64 = pil_to_base64(image2)
        print('[RM SERVER DEBUG] In call_vlm_2image(), the model used in the payload is:', VLLM_MODEL_PATH)
        print('[RM SERVER DEBUG] In call_vlm_2image(), the VLM temperature actually used is:', VLM_TEMPERATURE)
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": VLLM_MODEL_PATH,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image1_b64}},
                        {"type": "image_url", "image_url": {"url": image2_b64}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            "temperature": VLM_TEMPERATURE,
            "max_tokens": 1024
        }

        response = requests.post(
            f"{VLLM_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            answer = result["choices"][0]["message"]["content"].strip().lower()
            return answer
        else:
            return "no"

    except Exception as e:
        print(f"[VLM 2-Image Call Error] {e}")
        return "no"


# ========== Task Type Validation Functions ==========

def _calculate_iou(box1, box2):
    """Compute the IoU of two bounding boxes"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    return intersection / union


def _count_object(boxes, iou_threshold=0.8):
    """Count the number of objects after de-duplication"""
    if len(boxes) == 0:
        return 0
    
    boxes = boxes.copy()
    
    while True:
        if len(boxes) <= 1:
            return len(boxes)
        
        boxes_removed = False
        i = 0
        while i < len(boxes):
            j = i + 1
            while j < len(boxes):
                if _calculate_iou(boxes[i], boxes[j]) > iou_threshold:
                    boxes.pop(j)
                    boxes_removed = True
                else:
                    j += 1
            i += 1
        
        if not boxes_removed:
            return len(boxes)


VLM_ADD_PROMPT = """The first image is the original, and the second image reflects the changes made according to the editing instruction in subject addition. Can you determine if the editing instruction was successfully applied?
The editing instruction is: {instruction}

Please respond with "yes" or "no."""

VLM_REPLACE_PROMPT = """The first image is the original, and the second image reflects the changes made according to the editing instruction in subject replacement. Can you determine if the editing instruction was successfully applied?
The editing instruction is: {instruction}

Please respond with "yes" or "no."""

VLM_COLOR_PROMPT = """Look at the object in the image. Is the {object_name} {new_color}? Please answer only 'YES' or 'NO'."""

VLM_MATERIAL_PROMPT = """Is it possible that the {object_name} is made of {new_material}? Please answer only 'YES' or 'NO'."""

VLM_TEXT_PROMPT = """What text do you see in this image? Output only the text content, nothing else."""

VLM_BACKGROUND_PROMPT = """Look at the background of this image. Does the background show [{background}]? Please answer only 'YES' or 'NO'."""

VLM_STRICT_COUNT_PROMPT = """You are asked to count the number of an object. Please answer only the number. 
For example, if there are 3 dogs, you should answer '3'. If there is no dog, you should answer '0'.

Object name: {object_name}
Number of objects:"""


def parse_instruction(task_type, instruction):
    """Parse the instruction and extract the key information"""
    import re
    
    if task_type == "subject_replace":
        pattern = r"Replace \[([^\]]+)\] with \[([^\]]+)\]"
        match = re.search(pattern, instruction)
        if match:
            return [match.group(1), match.group(2)]
    
    elif task_type == "subject_remove":
        pattern = r"Remove \[([^\]]+)\]"
        match = re.search(pattern, instruction)
        if match:
            return [match.group(1)]
    
    elif task_type == "material_alter":
        pattern = r"Change the material of \[([^\]]+)\] to \[([^\]]+)\]"
        match = re.search(pattern, instruction)
        if match:
            return [match.group(1), match.group(2)]
    
    elif task_type == "color_alter":
        pattern = r"Change the color of \[([^\]]+)\] to \[([^\]]+)\]"
        match = re.search(pattern, instruction)
        if match:
            return [match.group(1), match.group(2)]
    
    elif task_type == "subject_add":
        pattern1 = r"Add \[([^\]]+)\] on the \[([^\]]+)\] of \[([^\]]+)\]"
        match1 = re.search(pattern1, instruction)
        if match1:
            return [match1.group(1), match1.group(2), match1.group(3)]
        
        pattern2 = r"Add \[([^\]]+)\]"
        match2 = re.search(pattern2, instruction)
        if match2:
            return [match2.group(1)]
    
    elif task_type == "text_change":
        pattern1 = r"Replace the text '\[([^\]]+)\]' on \[([^\]]+)\] with '\[([^\]]+)\]'"
        match1 = re.search(pattern1, instruction)
        if match1:
            return [match1.group(1), match1.group(2), match1.group(3)]
        
        pattern2 = r"Add text '\[([^\]]+)\]' on the image"
        match2 = re.search(pattern2, instruction)
        if match2:
            return [match2.group(1)]
    
    elif task_type == "position_change":
        pattern = r"Change the position of \[([^\]]+)\] to \[([^\]]+)\] of \[([^\]]+)\]"
        match = re.search(pattern, instruction)
        if match:
            return [match.group(1), match.group(2), match.group(3)]
    
    elif task_type == "count_change":
        pattern = r"Change the count of \[([^\]]+)\] to \[([^\]]+)\]"
        match = re.search(pattern, instruction)
        if match:
            return [match.group(1), match.group(2)]
    
    elif task_type == "background_change":
        pattern1 = r"Change the background to ([^,]+), remain the (.+) unchanged"
        match1 = re.search(pattern1, instruction)
        if match1:
            return [match1.group(1)] + match1.group(2).split(",")
        
        pattern2 = r"Change the background to \[([^\]]+)\]"
        match2 = re.search(pattern2, instruction)
        if match2:
            return [match2.group(1)]
    
    return None


def evaluate_single(ref_img: Image.Image, gen_img: Image.Image, 
                    prompt: str, formatted_instruction: str, task_type: str) -> Tuple[float, str]:
    """
    Evaluate instruction following for a single image

    Returns:
        (score, reason): score is 0 or 1
    """
    position_threshold = 0.03
    print('Entering evaluate_single() in edival_reward_server.py...')
    try:
        # ========== Subject Add ==========
        if task_type == "subject_add":
            parsed = parse_instruction("subject_add", formatted_instruction)
            if parsed is None:
                return 0, f"Invalid instruction format: {formatted_instruction}"
            
            if len(parsed) == 1:
                new_object = parsed[0]
                #
                vlm_resp = call_vlm_2image(ref_img, gen_img, 
                    VLM_ADD_PROMPT.format(instruction=formatted_instruction))
                #
                print('[VLM OUTPUT DEBUG] In edival_reward_server.py, function evaluate_single(), vlm_resp = ', vlm_resp)
                if "yes" not in vlm_resp:
                    return 0, f"VLM pre-check failed. Response: '{vlm_resp}'"
                
                # GroundingDINO detection
                detections = call_groundingdino(gen_img, new_object, return_all=True)
                if len(detections.get("score", [])) > 0:
                    return 1, f"Successfully added [{new_object}]. Detected {len(detections['score'])} instance(s)."
                else:
                    return 0, f"Failed to add [{new_object}]. No instances detected."
                    
            elif len(parsed) == 3:
                new_object, position, ref_object = parsed
                
                # VLM pre-check
                vlm_resp = call_vlm_2image(ref_img, gen_img,
                    VLM_ADD_PROMPT.format(instruction=formatted_instruction))
                print('[VLM OUTPUT DEBUG] In edival_reward_server.py, function evaluate_single(), vlm_resp = ', vlm_resp)
                if "yes" not in vlm_resp:
                    return 0, f"VLM pre-check failed. Response: '{vlm_resp}'"
                
                # Detect the reference object and the target object
                src_ref = call_groundingdino(ref_img, ref_object, return_all=True)
                target_obj = call_groundingdino(gen_img, new_object, return_all=True)
                print(f'>>>In evaluate_single(), call_groundingdino succeeded, returned src_ref={src_ref}, target_obj={target_obj}')
                
                if len(src_ref.get("score", [])) == 0 or len(target_obj.get("score", [])) == 0:
                    return 0, f"Failed to detect objects. Ref: {len(src_ref.get('score', []))}, Target: {len(target_obj.get('score', []))}"
                
                ref_centers = src_ref.get("center", [])
                target_centers = target_obj.get("center", [])
                
                for ref_center in ref_centers:
                    ref_norm = [ref_center[0] / ref_img.size[0], ref_center[1] / ref_img.size[1]] 
                    for target_center in target_centers:
                        target_norm = [target_center[0] / gen_img.size[0], target_center[1] / gen_img.size[1]] # normalize here
                        if position == "left" and target_norm[0] - ref_norm[0] < -position_threshold:
                            return 1, f"Successfully added [{new_object}] to the left of [{ref_object}]"
                        elif position == "right" and target_norm[0] - ref_norm[0] > position_threshold:
                            return 1, f"Successfully added [{new_object}] to the right of [{ref_object}]"
                        elif position == "above" and target_norm[1] - ref_norm[1] < -position_threshold:
                            return 1, f"Successfully added [{new_object}] above [{ref_object}]"
                        elif position == "below" and target_norm[1] - ref_norm[1] > position_threshold:
                            return 1, f"Successfully added [{new_object}] below [{ref_object}]"
                
                return 0, f"Failed to add [{new_object}] to the {position} of [{ref_object}]"
        
        # ========== Subject Remove ==========
        elif task_type == "subject_remove":
            parsed = parse_instruction("subject_remove", formatted_instruction)
            if parsed is None:
                return 0, f"Invalid instruction format: {formatted_instruction}"
            
            object_name = parsed[0]
            
            # Detect the object in the source image
            src_obj = call_groundingdino(ref_img, object_name, threshold=0.35, delete_large_box=True)
            if len(src_obj.get("score", [])) == 0:
                return 0, f"Failed to detect [{object_name}] in source image"
            
            src_box = src_obj["box"][0]
            
            # Detect in the target image
            target_obj = call_groundingdino(gen_img, object_name, threshold=0.35, delete_large_box=True)
            
            if len(target_obj.get("score", [])) == 0:
                return 1, f"Successfully removed [{object_name}]"
            
            # Check whether it's the same object (IoU > 0.2)
            if _calculate_iou(src_box, target_obj["box"][0]) < 0.2:
                return 1, f"Successfully removed [{object_name}] (different location)"
            
            return 0, f"Failed to remove [{object_name}]"
        
        # ========== Subject Replace ==========
        elif task_type == "subject_replace":
            parsed = parse_instruction("subject_replace", formatted_instruction)
            if parsed is None:
                return 0, f"Invalid instruction format: {formatted_instruction}"
            
            object_name, new_object = parsed
            
            # VLM pre-check
            vlm_resp = call_vlm_2image(ref_img, gen_img,
                VLM_REPLACE_PROMPT.format(instruction=formatted_instruction))
            if "yes" not in vlm_resp:
                return 0, f"VLM pre-check failed. Response: '{vlm_resp}'"
            
            # Strip plural suffix
            object_name = object_name[:-2] if object_name.endswith('es') else object_name[:-1] if object_name.endswith('s') else object_name
            new_object = new_object[:-2] if new_object.endswith('es') else new_object[:-1] if new_object.endswith('s') else new_object
            
            # Detect both objects
            src_obj = call_groundingdino(ref_img, object_name, return_all=True)
            target_obj = call_groundingdino(gen_img, new_object, return_all=True)
            
            if len(src_obj.get("score", [])) == 0 or len(target_obj.get("score", [])) == 0:
                return 0, f"Failed to detect objects. Source: {len(src_obj.get('score', []))}, Target: {len(target_obj.get('score', []))}"
            
            # Check for overlap
            for src_box in src_obj.get("box", []):
                for target_box in target_obj.get("box", []):
                    if _calculate_iou(src_box, target_box) > 0:
                        return 1, f"Successfully replaced [{object_name}] with [{new_object}]"
            
            return 0, f"Failed to replace [{object_name}] with [{new_object}]"
        
        # ========== Color Alter ==========
        elif task_type == "color_alter":
            parsed = parse_instruction("color_alter", formatted_instruction)
            if parsed is None:
                return 0, f"Invalid instruction format: {formatted_instruction}"
            
            object_name, new_color = parsed
            
            vlm_resp = call_vlm_2image(ref_img, gen_img,
                f"Is the {object_name} now {new_color}? Answer YES or NO only.")
            
            if "yes" in vlm_resp:
                return 1, f"Successfully changed color of [{object_name}] to [{new_color}]"
            else:
                return 0, f"Failed to change color of [{object_name}] to [{new_color}]. VLM: '{vlm_resp}'"
        
        # ========== Material Alter ==========
        elif task_type == "material_alter":
            parsed = parse_instruction("material_alter", formatted_instruction)
            if parsed is None:
                return 0, f"Invalid instruction format: {formatted_instruction}"
            
            object_name, new_material = parsed
            
            vlm_resp = call_vlm_2image(ref_img, gen_img,
                f"Is the {object_name} now made of {new_material}? Answer YES or NO only.")
            
            if "yes" in vlm_resp:
                return 1, f"Successfully changed material of [{object_name}] to [{new_material}]"
            else:
                return 0, f"Failed to change material of [{object_name}] to [{new_material}]. VLM: '{vlm_resp}'"
        
        # ========== Text Change ==========
        elif task_type == "text_change":
            parsed = parse_instruction("text_change", formatted_instruction)
            if parsed is None:
                return 0, f"Invalid instruction format: {formatted_instruction}"
            
            if len(parsed) == 3:
                existing_text, object_name, new_text = parsed
                
                # Detect the object
                src_obj = call_groundingdino(ref_img, object_name)
                if len(src_obj.get("score", [])) == 0:
                    return 0, f"Failed to detect [{object_name}] in source image"
                
                # Crop and query the VLM
                # Simplified handling: directly ask the VLM whether it sees the new text
                vlm_resp = call_vlm(gen_img, VLM_TEXT_PROMPT)
                
                if new_text.lower() in vlm_resp:
                    return 1, f"Successfully changed text to '[{new_text}]'. VLM detected: '{vlm_resp}'"
                else:
                    return 0, f"Failed to change text to '[{new_text}]'. VLM detected: '{vlm_resp}'"
                    
            elif len(parsed) == 1:
                new_text = parsed[0]
                vlm_resp = call_vlm(gen_img, VLM_TEXT_PROMPT)
                
                if new_text.lower() in vlm_resp:
                    return 1, f"Successfully added text '[{new_text}]'. VLM detected: '{vlm_resp}'"
                else:
                    return 0, f"Failed to add text '[{new_text}]'. VLM detected: '{vlm_resp}'"
        
        # ========== Position Change ==========
        elif task_type == "position_change":
            parsed = parse_instruction("position_change", formatted_instruction)
            if parsed is None:
                return 0, f"Invalid instruction format: {formatted_instruction}"
            
            target_object, position, reference_object = parsed
            
            # Detect all objects
            src_ref = call_groundingdino(ref_img, reference_object, return_all=True, threshold=0.4)
            src_target = call_groundingdino(ref_img, target_object, return_all=True, threshold=0.4)
            target_ref = call_groundingdino(gen_img, reference_object, return_all=True, threshold=0.4)
            target_target = call_groundingdino(gen_img, target_object, return_all=True, threshold=0.4)
            
            # Check whether the counts match
            src_ref_count = _count_object(src_ref.get("box", []))
            target_ref_count = _count_object(target_ref.get("box", []))
            src_target_count = _count_object(src_target.get("box", []))
            target_target_count = _count_object(target_target.get("box", []))
            
            if src_ref_count != target_ref_count:
                return 0, f"Reference object count changed from {src_ref_count} to {target_ref_count}"
            
            if src_target_count != target_target_count:
                return 0, f"Target object count changed from {src_target_count} to {target_target_count}"
            
            # Check the positional relationship in the target image
            if len(target_ref.get("score", [])) > 0 and len(target_target.get("score", [])) > 0:
                ref_center = target_ref["center"][0]
                target_center = target_target["center"][0]
                
                if position == "left" and target_center[0] < ref_center[0]:
                    return 1, f"Successfully moved [{target_object}] to the left of [{reference_object}]"
                elif position == "right" and target_center[0] > ref_center[0]:
                    return 1, f"Successfully moved [{target_object}] to the right of [{reference_object}]"
                elif position == "above" and target_center[1] < ref_center[1]:
                    return 1, f"Successfully moved [{target_object}] above [{reference_object}]"
                elif position == "below" and target_center[1] > ref_center[1]:
                    return 1, f"Successfully moved [{target_object}] below [{reference_object}]"
                else:
                    return 0, f"Failed to move [{target_object}] to the {position} of [{reference_object}]"
            
            return 0, f"Failed to detect objects in target image"
        
        # ========== Count Change ==========
        elif task_type == "count_change":
            parsed = parse_instruction("count_change", formatted_instruction)
            if parsed is None:
                return 0, f"Invalid instruction format: {formatted_instruction}"
            
            object_name, target_count = parsed
            
            # Detect in the source image and the target image
            src_obj = call_groundingdino(ref_img, object_name)
            target_obj = call_groundingdino(gen_img, object_name, return_all=True)
            
            actual_count = _count_object(target_obj.get("box", []))
            
            if len(src_obj.get("score", [])) > 0 and int(actual_count) == int(target_count):
                return 1, f"Successfully changed count of [{object_name}] to {target_count}"
            else:
                return 0, f"Failed to change count of [{object_name}] to {target_count}. Actual count: {actual_count}"
        
        # ========== Background Change ==========
        elif task_type == "background_change":
            parsed = parse_instruction("background_change", formatted_instruction)
            if parsed is None:
                return 0, f"Invalid instruction format: {formatted_instruction}"
            
            background = parsed[0]
            remain_objects = parsed[1:] if len(parsed) > 1 else []
            
            # Check whether the objects that should remain are still present
            for obj in remain_objects:
                obj_det = call_groundingdino(gen_img, obj, threshold=0.25)
                if len(obj_det.get("score", [])) == 0:
                    return 0, f"Failed to detect remaining object [{obj}]"
                # Check if it's overly large (possible false positive)
                box = obj_det["box"][0]
                if abs(box[2] - box[0]) > 0.9 and abs(box[3] - box[1]) > 0.9:
                    return 0, f"False positive detection of [{obj}]"
            
            # VLM check on the background
            vlm_resp = call_vlm(gen_img, VLM_BACKGROUND_PROMPT.format(background=background))
            
            if "yes" in vlm_resp:
                return 1, f"Successfully changed background to [{background}]"
            else:
                return 0, f"Failed to change background to [{background}]. VLM: '{vlm_resp}'"
        
        else:
            return 0, f"Unknown task type: {task_type}"
            
    except Exception as e:
        error_msg = f"Error evaluating {task_type}: {str(e)}"
        print(f"[Evaluation Error] {error_msg}")
        traceback.print_exc()
        return 0, error_msg


@app.route("/mode/instruction_following", methods=["POST"])
def inference_instruction_following():
    """EDiVal Instruction Following evaluation endpoint"""
    try:
        # Parse the request data
        data = pickle.loads(request.get_data())
        
        ref_image_bytes_list = data.get("ref_images", [])
        image_bytes_list = data["images"]
        prompts = data["prompts"]
        metadatas = data.get("metadatas", [])
        
        # Validate the input
        if len(image_bytes_list) != len(prompts) or len(prompts) != len(metadatas):
            error_msg = f"Length mismatch: images({len(image_bytes_list)}), prompts({len(prompts)}), metadatas({len(metadatas)})"
            response = pickle.dumps({"error": error_msg})
            return response, 400
        
        # If ref_images is not provided, use placeholders
        if not ref_image_bytes_list:
            ref_image_bytes_list = [b""] * len(image_bytes_list)
        
        # Convert to PIL Images
        ref_images = []
        gen_images = []
        
        for ref_bytes in ref_image_bytes_list:
            if ref_bytes:
                ref_images.append(Image.open(BytesIO(ref_bytes)).convert("RGB"))
            else:
                ref_images.append(None)
        
        for img_bytes in image_bytes_list:
            gen_images.append(Image.open(BytesIO(img_bytes)).convert("RGB"))
        
        # Batch evaluation (parallel via thread pool)
        futures = [
            executor.submit(
                evaluate_single, ref_img, gen_img, prompt,
                metadata.get("formatted_instruction", ""),
                metadata.get("task_type", "")
            )
            for ref_img, gen_img, prompt, metadata
            in zip(ref_images, gen_images, prompts, metadatas)
        ]

        scores = []
        reasons = []
        for future in futures:
            score, reason = future.result()
            scores.append(score)
            reasons.append(reason)
        
        # Return the result
        response = pickle.dumps({
            "scores": scores,
            "reasons": reasons
        })
        return response, 200
        
    except KeyError as e:
        error_msg = f"Missing required field: {str(e)}"
        print(f"[Server] Error: {error_msg}")
        response = pickle.dumps({"error": error_msg})
        return response, 400
        
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"[Server] Error:\n{error_msg}")
        response = pickle.dumps({"error": str(e)})
        return response, 500


@app.route("/mode/consistency", methods=["POST"])
def inference_consistency():
    """EDiVal Consistency evaluation endpoint

    Request body (pickle-serialized):
    {
        "base_images": [bytes, bytes, ...],     # Base images (JPEG bytes)
        "target_images": [bytes, bytes, ...],   # Target images (JPEG bytes)
        "metadatas": [                           # Metadata, must include
            {
                "unchanged_objects": List[str],  # List of unchanged objects
                "all_objects": List[str]         # List of all objects
            },
            ...
        ]
    }

    Response body (pickle-serialized):
    {
        "object_results": [dict, dict, ...],      # Object consistency results
        "background_results": [dict, dict, ...]   # Background consistency results
    }
    """
    try:
        # Parse the request data
        data = pickle.loads(request.get_data())

        base_image_bytes_list = data.get("base_images", [])
        target_image_bytes_list = data.get("target_images", [])
        metadatas = data.get("metadatas", [])

        # Validate the input
        if len(base_image_bytes_list) != len(target_image_bytes_list):
            error_msg = f"Length mismatch: base_images({len(base_image_bytes_list)}), target_images({len(target_image_bytes_list)})"
            response = pickle.dumps({"error": error_msg})
            return response, 400

        if len(base_image_bytes_list) != len(metadatas):
            error_msg = f"Length mismatch: images({len(base_image_bytes_list)}), metadatas({len(metadatas)})"
            response = pickle.dumps({"error": error_msg})
            return response, 400

        # Convert to PIL Images
        base_images = []
        target_images = []

        for img_bytes in base_image_bytes_list:
            base_images.append(Image.open(BytesIO(img_bytes)).convert("RGB"))

        for img_bytes in target_image_bytes_list:
            target_images.append(Image.open(BytesIO(img_bytes)).convert("RGB"))

        # Batch evaluation (parallel via thread pool)
        print('[RM SERVER DEBUG] In inference_consistency(), entering the thread-pool parallel batch evaluation...')
        futures = [
            executor.submit(
                evaluate_consistency_single,
                base_img, target_img,
                metadata.get("unchanged_objects", []),
                metadata.get("all_objects", [])
            )
            for base_img, target_img, metadata
            in zip(base_images, target_images, metadatas)
        ]

        object_results = []
        background_results = []
        for future in futures:
            obj_result, bg_result = future.result()
            object_results.append(obj_result)
            background_results.append(bg_result)

        # Return the result
        response = pickle.dumps({
            "object_results": object_results,
            "background_results": background_results
        })
        return response, 200

    except KeyError as e:
        error_msg = f"Missing required field: {str(e)}"
        print(f"[Server] Error: {error_msg}")
        response = pickle.dumps({"error": error_msg})
        return response, 400

    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"[Server] Error:\n{error_msg}")
        response = pickle.dumps({"error": str(e)})
        return response, 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        # Check dependent services
        g_dino_health = requests.get(f"{GROUNDINGDINO_URL}/health", timeout=5).json()
        g_dino_ok = g_dino_health.get("model_loaded", False)
    except:
        g_dino_ok = False

    try:
        # Simple check of whether vLLM is reachable (via the models endpoint)
        vlm_resp = requests.get(f"{VLLM_URL}/models", timeout=5)
        vlm_ok = vlm_resp.status_code == 200
    except:
        vlm_ok = False

    # Check DINOv3
    dinov3_ok = DINOV3_BUNDLE is not None

    status = "healthy" if (g_dino_ok and vlm_ok) else "degraded"

    return pickle.dumps({
        "status": status,
        "groundingdino": "ok" if g_dino_ok else "error",
        "vlm": "ok" if vlm_ok else "error",
        "dinov3": "ok" if dinov3_ok else "error",
        "services": {
            "groundingdino_url": GROUNDINGDINO_URL,
            "vlm_url": VLLM_URL
        }
    }), 200


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EDiVal Instruction Following Reward Server")
    parser.add_argument("--port", type=int, default=None,
                        help="Listen port (default: env var EDIVAL_PORT or 12342)")
    parser.add_argument("--vllm-url", type=str, default=None,
                        help="vLLM service URL (default: env var VLLM_URL or http://localhost:8000/v1)")
    parser.add_argument("--grounding-dino-url", type=str, default=None,
                        help="GroundingDINO service URL (default: env var GROUNDINGDINO_URL or http://localhost:12343)")
    parser.add_argument("--vlm-temperature", type=float, default=None,
                        help="VLM sampling temperature (default: env var VLM_TEMPERATURE or 0.0)")
    args = parser.parse_args()

    # Command-line args take highest priority and override the globals (env vars were already read at module load time)
    if args.vllm_url:
        VLLM_URL = args.vllm_url
    if args.grounding_dino_url:
        GROUNDINGDINO_URL = args.grounding_dino_url
    if args.vlm_temperature is not None:
        VLM_TEMPERATURE = args.vlm_temperature

    listen_port = args.port or int(os.getenv("EDIVAL_PORT", "12342"))

    print("="*60)
    print("EDiVal Instruction Following Reward Server")
    print("="*60)
    print(f"Listen port:       {listen_port}")
    print(f"GroundingDINO URL: {GROUNDINGDINO_URL}")
    print(f"VLLM URL:          {VLLM_URL}")
    print(f"VLM Temperature:   {VLM_TEMPERATURE}")
    dinov3_status = "loaded" if DINOV3_BUNDLE is not None else "not loaded"
    print(f"DINOv3 Model:      {dinov3_status}")
    print("\nPlease ensure dependent services are running:")
    print("  bash start_servers.sh        # starts vLLM")
    print("  bash start_groundingdino.sh  # starts GroundingDINO")
    print("\n" + "="*60)
    print(f"Starting Flask server on 0.0.0.0:{listen_port}...")
    print("API endpoints:")
    print("  POST /mode/instruction_following")
    print("  POST /mode/consistency")
    print("  GET  /health")
    print("="*60)

    app.run(host="0.0.0.0", port=listen_port, debug=False, threaded=True)