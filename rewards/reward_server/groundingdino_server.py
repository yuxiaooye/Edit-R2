"""
GroundingDINO Detection Server

Standalone GroundingDINO object detection service.

Usage:
    python groundingdino_server.py

API:
    POST /detect
    Request body (JSON):
    {
        "image": base64_encoded_image,  # JPEG/PNG base64
        "target_object": str,           # target object name
        "threshold": float (optional),  # detection threshold, default 0.3
        "return_all": bool (optional)   # whether to return all detections, default False
    }
    
    Response body (JSON):
    {
        "success": bool,
        "detections": {
            "label": [...],
            "score": [...],
            "box": [...],       # [x1, y1, x2, y2] normalized
            "center": [...]     # [cx, cy] pixel coordinates
        }
    }

    POST /health
    Response: {"status": "healthy", "model_loaded": true}
"""

import os
import sys
import threading
import torch
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify
import traceback
import numpy as np

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

# Global model
grounding_model = None

# GroundingDINO internally writes features to self.features via a forward hook.
# This mutable shared state causes a race condition under concurrent forward() calls,
# resulting in AttributeError: 'GroundingDINO' object has no attribute 'features'.
# Use a global mutex to serialize inference.
_inference_lock = threading.Lock()


def init_python_path():
    """Initialize Python path"""
    server_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(server_dir)
    
    # Add GroundingDINO path
    groundingdino_path = os.path.join(project_root, 'GroundingDINO')
    if groundingdino_path not in sys.path:
        sys.path.insert(0, groundingdino_path)
    
    # Add detector path
    detector_path = os.path.join(project_root, 'detector')
    if detector_path not in sys.path:
        sys.path.insert(0, detector_path)
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def load_model_once():
    """Load GroundingDINO model"""
    global grounding_model
    
    if grounding_model is not None:
        return grounding_model
    
    try:
        from groundingdino.util.inference import load_model
        
        config_path = os.environ.get("GROUNDING_CONFIG_PATH")
        weights_path = os.environ.get("GROUNDING_WEIGHT_PATH")
        
        print(f"[GroundingDINO] Loading model from {weights_path}...")
        grounding_model = load_model(config_path, weights_path)
        print("[GroundingDINO] Model loaded successfully!")
        
        return grounding_model
        
    except Exception as e:
        print(f"[GroundingDINO] Error loading model: {e}")
        traceback.print_exc()
        raise


def detect_objects(image: Image.Image, target_object, threshold: float = 0.3, 
                   return_all: bool = False, delete_large_box: bool = False):
    """
    Detect objects using GroundingDINO.

    Args:
        target_object: str or List[str]. When a list is given, all object names are
                       joined with " . " into a single prompt for one-shot detection
                       (following the standard GroundingDINO usage).

    Returns:
        Dict with detected objects information
    """
    from groundingdino.util.inference import predict
    import groundingdino.datasets.transforms as T
    
    model = grounding_model
    if model is None:
        return {"label": [], "score": [], "box": [], "center": []}
    
    try:
        # Ensure PIL image is in RGB format
        if isinstance(image, Image.Image):
            image_pil = image.convert('RGB')
        else:
            image_pil = Image.fromarray(image).convert('RGB')
        
        # Apply DINO's transforms
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        image_tensor, _ = transform(image_pil, None)
        
        # Create text prompt — support both single string and list of strings
        if isinstance(target_object, list):
            cleaned = [obj.strip().lower().replace(".", "") for obj in target_object]
            text_prompt = " . ".join(cleaned) + " ."
        else:
            text_prompt = target_object.strip().lower().replace(".", "") + " ."
        
        # Predict with DINO — lock to prevent concurrent forward() from corrupting self.features
        print(f">>>[GroundingDINO] Predict with DINO — acquiring lock to serialize inference")
        with _inference_lock:
            boxes, logits, phrases = predict(
                model=model,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=threshold,
                text_threshold=threshold
            )
        
        # Convert results
        h, w = image_pil.size[1], image_pil.size[0]
        
        scores = []
        bbox_list = []
        centers = []
        labels = []
        
        for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
            # Convert normalized coordinates to pixel coordinates
            # Ensure box values are Python floats (not Tensors)
            if hasattr(box, 'item'):
                box = [b.item() for b in box]
            center_x, center_y, width, height = box
            center_x, center_y, width, height = center_x * w, center_y * h, width * w, height * h
            
            # Convert to x1, y1, x2, y2 format (normalized coordinates)
            x1 = float((center_x - width / 2) / w)
            y1 = float((center_y - height / 2) / h)
            x2 = float((center_x + width / 2) / w)
            y2 = float((center_y + height / 2) / h)
            
            if delete_large_box:
                if abs(x1 - x2) > 0.98 and abs(y1 - y2) > 0.98:
                    continue
            
            scores.append(float(logit.item() if hasattr(logit, 'item') else logit))
            bbox_list.append([x1, y1, x2, y2])
            centers.append([int(center_x), int(center_y)])
            labels.append(phrase)
        
        # Only return highest scoring detection if not return_all
        if scores:
            if return_all:
                return {
                    "label": labels,
                    "score": scores,
                    "box": bbox_list,
                    "center": centers
                }
            else:
                max_score_idx = scores.index(max(scores))
                return {
                    "label": [labels[max_score_idx]],
                    "score": [scores[max_score_idx]],
                    "box": [bbox_list[max_score_idx]],
                    "center": [centers[max_score_idx]]
                }
        else:
            return {"label": [], "score": [], "box": [], "center": []}
            
    except Exception as e:
        print(f"[GroundingDINO] Error in detection: {e}")
        traceback.print_exc()
        return {"label": [], "score": [], "box": [], "center": []}


@app.route("/detect", methods=["POST"])
def detect_endpoint():
    """Object detection endpoint"""
    try:
        data = request.get_json()
        
        # Parse request
        image_b64 = data.get("image", "")
        target_object = data.get("target_objects") or data.get("target_object", "")
        threshold = data.get("threshold", 0.3)
        return_all = data.get("return_all", False)
        delete_large_box = data.get("delete_large_box", False)
        
        if not image_b64 or not target_object:
            return jsonify({
                "success": False,
                "error": "Missing required fields: 'image' or 'target_object'/'target_objects'"
            }), 400
        
        # Decode base64 image
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        # Run detection
        detections = detect_objects(image, target_object, threshold, return_all, delete_large_box)
        
        return jsonify({
            "success": True,
            "detections": detections
        })
        
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"[GroundingDINO Server] Error: {error_msg}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check"""
    model_loaded = grounding_model is not None
    return jsonify({
        "status": "healthy" if model_loaded else "loading",
        "model_loaded": model_loaded
    })


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GroundingDINO Detection Worker")
    parser.add_argument("--port", type=int, default=12343, help="Listening port (default 12343)")
    parser.add_argument("--gpu-id", type=str, default=None,
                        help="GPU ID to bind, overrides CUDA_VISIBLE_DEVICES (e.g. '0', '1')")
    args = parser.parse_args()

    # Set visible devices before CUDA context is initialized
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        print(f"[GroundingDINO] CUDA_VISIBLE_DEVICES set to {args.gpu_id}")

    print("="*60)
    print(f"GroundingDINO Detection Worker  (port={args.port})")
    print("="*60)

    # Initialize paths
    init_python_path()

    # Load model
    try:
        load_model_once()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        sys.exit(1)

    print(f"\nStarting server on 0.0.0.0:{args.port}...")
    print("API endpoint: POST /detect")
    print("Health check: GET /health")
    print("="*60)

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
