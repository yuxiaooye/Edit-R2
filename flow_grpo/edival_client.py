import os
import sys
import re
import time
import base64
import pickle
import requests
from requests.adapters import HTTPAdapter, Retry
from concurrent.futures import ThreadPoolExecutor as _ClientThreadPool
from typing import List, Dict, Tuple, Optional
import torch
import numpy as np
from PIL import Image
from io import BytesIO


DEFAULT_SERVER_URL = "http://100.99.129.57:12342"
SERVER_URL = os.getenv("EDIVAL_SERVER_URL", DEFAULT_SERVER_URL)
BATCH_SIZE = 64


DEFAULT_GA_VLM_URL = "http://localhost:8000/v1"
GA_VLM_URL = os.getenv("EDIVAL_GA_VLM_URL", DEFAULT_GA_VLM_URL)
GA_MODEL_PATH = os.getenv("VLLM_MODEL_PATH", "/path/to/Qwen2.5-VL-32B-Instruct")
GA_VLM_TEMPERATURE = float(os.getenv("EDIVAL_GA_TEMPERATURE", "0.6"))

MULTITURN_VLM_PROMPT = """你是一个专门评估多轮图像编辑任务的专家。你将获得一个由N张图像和N-1条编辑指令组成的序列。

**任务定义：**
该任务包含N-1个连续的编辑轮次。
- **第 i 轮**（i 从1到 N-1）：
  - **输入：** 图像 i
  - **指令：** 指令 i
  - **输出：** 图像 i+1

你的目标是从第1轮开始，按顺序评估每一轮编辑。

**每轮的评估标准：**
对于每一轮 i，根据指令 i 比较图像 i（输入）和图像 i+1（输出）。只有当一轮**同时满足**以下所有标准时，才算成功（"yes"）：
1. **指令遵循：** 图像 i+1 成功反映了指令 i 所要求的更改。
2. **全局约束遵循：** 如果第一个指令设定了影响整个会话的全局约束（例如，"后续编辑中添加或修改的对象必须是黄色"等），那么图像 i+1 必须遵循这个约束。具体规则如下：
     - **作用范围：** 全局约束的遵循仅需体现在当前轮编辑所涉及的物体上。例如当指令为"在桌子上添加一个杯子"且全局约束为"之后的编辑内容都应该是玻璃材质"时，那么成功的图像i+1上应该只有新添加的杯子是玻璃材质，而不能把桌子或其他物体也变成玻璃材质。
     - **移除类指令豁免：** 如果当前轮的指令是移除/删除某个物体（例如"remove the cup"、"去掉背景中的树"等），则该轮无需考虑全局约束——无论编辑结果如何，都视为全局约束被成功遵循。因为移除操作不涉及添加或修改物体，全局约束自然不适用。
     - **禁止整体色调替代：** 全局约束要求的是对编辑涉及的具体物体施加约束，而不是对图片整体进行色调变换。如果编辑后的图片只是将整张图片的色调转变为全局约束所规定的颜色（例如全局约束要求"黄色"，而图片整体被加上了黄色滤镜），但编辑涉及的具体物体并未真正体现该约束，则应判定为全局约束未被成功遵循。

**执行与输出逻辑：**
逐一评估各轮（第1轮，第2轮，...）。

- **如果第 i 轮成功（"yes"）：**
  输出：
  <answer_turn_i> yes </answer_turn_i>
  <reason_turn_i> 简要解释成功的原因。 </reason_turn_i>
  然后继续评估第 i+1 轮（如果存在）。

- **如果第 i 轮失败（"no"）：**
  输出：
  <answer_turn_i> no </answer_turn_i>
  <reason_turn_i> 对失败原因的解释（例如，"未能添加对象" 或 "成功添加了对象B，但意外删除了对象A"）。 </reason_turn_i>
  **立即停止。** 不要评估任何后续轮次。
  输出： <answer_final> no </answer_final>

- **如果所有轮次（1 到 N-1）都成功：**
  在评估完最后一轮后，输出： <answer_final> yes </answer_final>

**输入数据：**
**图像序列：**
- 图像 1: 初始图像
- 图像 2: 第1轮的结果
...
- 图像 N: 第N-1轮的结果

**指令：**
{instructions_formatted}

**回复格式：**
请严格按照上述类似XML的标签提供你的评估。不要在标签之外包含任何对话性文本。将标签中的 'i' 替换为实际的轮次编号（例如，<answer_turn_1>, <answer_turn_2>）。
"""

MULTITURN_CU_VLM_PROMPT = """你是一个专门评估多轮图像编辑任务中"内容理解（Content Understanding）"能力的专家。你将获得一个由N张图像和N-1条编辑指令组成的序列。

**任务背景：**
在多轮图像编辑的真实场景中，用户在建立对象上下文后，会自然地从使用完整对象名称过渡到使用代词（如"it"、"them"、"there"、"its"等）来指代前轮编辑过的对象。这要求编辑模型能够准确地进行"代词消解"——即理解代词指代的具体对象，并对该对象执行正确的编辑操作。

**任务定义：**
该任务包含N-1个连续的编辑轮次。
- **第 i 轮**（i 从1到 N-1）：
  - **输入：** 图像 i
  - **模型接收的指令：** 指令 i（可能包含代词，如"it"、"them"、"there"、"its"）
  - **显式参照指令：** 格式化指令 i（用方括号标明了代词所指代的具体对象，作为评估的客观参照）
  - **输出：** 图像 i+1

你的目标是从第1轮开始，按顺序评估每一轮编辑。

**每轮的评估标准：**
对于每一轮 i，根据指令 i 比较图像 i（输入）和图像 i+1（输出）。只有当一轮**同时满足**以下所有标准时，才算成功（"yes"）：

1. **代词消解正确（Content Understanding）：** 如果指令 i 中包含代词（如"it"、"them"、"there"、"its"等），图像 i+1 中被编辑的对象必须与格式化指令 i 中方括号内标注的对象一致。也就是说，模型必须正确理解代词指代的是哪个具体对象，并对该对象（而非其他对象）执行了编辑操作。具体规则如下：
   - **"it"/"them" 指代：** 代词指代前轮操作过的同一对象。例如，如果指令是"Remove it"，格式化指令是"Remove [red car]"，那么图像 i+1 中被移除的应该是红色汽车，而非其他对象。
   - **"its" 指代：** 所有格代词指代前轮操作过的对象的属性。例如，如果指令是"Change its color to blue"，格式化指令是"Change the color of [wooden brown door] to [blue]"，那么图像 i+1 中颜色变蓝的应该是木质棕色门。
   - **"there" 指代：** 空间代词指代前轮中某对象被移除后的位置。例如，如果指令是"Add a bench there"，格式化指令是"Add [bench] on the [left] of [flower bed]"，那么图像 i+1 中应该在之前花坛所在的位置添加了长椅。
   - **第1轮通常不含代词**（因为没有前文上下文），此时仅评估下述"指令遵循"标准。

2. **指令遵循（Instruction Following）：** 图像 i+1 成功反映了指令 i 所要求的更改——包括编辑类型正确（添加/删除/替换/颜色修改等）以及属性值正确（如颜色确实变为指定颜色）。

**执行与输出逻辑：**
逐一评估各轮（第1轮，第2轮，...）。

- **如果第 i 轮成功（"yes"）：**
  输出：
  <answer_turn_i> yes </answer_turn_i>
  <reason_turn_i> 简要解释成功的原因。 </reason_turn_i>
  然后继续评估第 i+1 轮（如果存在）。

- **如果第 i 轮失败（"no"）：**
  输出：
  <answer_turn_i> no </answer_turn_i>
  <reason_turn_i> 对失败原因的详细解释。请明确指出是代词消解错误还是指令遵循错误，例如：
    - 代词消解错误："代词'it'应指代[red car]，但模型错误地移除了蓝色卡车"
    - 指令遵循错误："正确识别了目标对象[wooden door]，但颜色未能成功修改为蓝色"
    - 两者皆错："代词'its'应指代[silver helmet]的属性，但模型修改了其他对象的颜色，且颜色也不正确" </reason_turn_i>
  **立即停止。** 不要评估任何后续轮次。
  输出： <answer_final> no </answer_final>

- **如果所有轮次（1 到 N-1）都成功：**
  在评估完最后一轮后，输出： <answer_final> yes </answer_final>

**输入数据：**
**图像序列：**
- 图像 1: 初始图像
- 图像 2: 第1轮的结果
...
- 图像 N: 第N-1轮的结果

**指令：**
{instructions_formatted}

**回复格式：**
请严格按照上述类似XML的标签提供你的评估。不要在标签之外包含任何对话性文本。将标签中的 'i' 替换为实际的轮次编号（例如，<answer_turn_1>, <answer_turn_2>）。
"""


def _create_session():
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=False
    )
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def _pil_to_bytes(image: Image.Image, format: str = "JPEG") -> bytes:
    """Convert a PIL Image to bytes."""
    buffer = BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def evaluate_via_server(
    ref_images: List[Image.Image],
    images: List[Image.Image],
    prompts: List[str],
    metadatas: List[Dict],
    server_url: Optional[str] = None,
    timeout: int = 180
) -> Tuple[List[float], List[str]]:
    url = server_url or SERVER_URL
    endpoint = f"{url}/mode/instruction_following"
    

    # Convert images to bytes
    ref_bytes_list = [_pil_to_bytes(img) for img in ref_images]
    img_bytes_list = [_pil_to_bytes(img) for img in images]

    # Prepare request payload
    data = {
        "ref_images": ref_bytes_list,
        "images": img_bytes_list,
        "prompts": prompts,
        "metadatas": metadatas
    }

    # Send request
    session = _create_session()
    try:
        response = session.post(
            endpoint,
            data=pickle.dumps(data),
            timeout=timeout
        )
        response.raise_for_status()

        # Parse response
        result = pickle.loads(response.content)

        if "error" in result:
            raise RuntimeError(f"Server error: {result['error']}")

        return result["scores"], result["reasons"]

    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(
            f"Cannot connect to EDiVal server at {url}. "
            f"Please ensure the server is running: "
            f"python reward_server/edival_reward_server.py"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Error calling EDiVal server: {e}") from e


def evaluate_consistency_via_server(
    base_images: List[Image.Image],
    target_images: List[Image.Image],
    metadatas: List[Dict],
    server_url: Optional[str] = None,
    timeout: int = 180
) -> Tuple[List[Dict], List[Dict]]:

    url = server_url or SERVER_URL
    endpoint = f"{url}/mode/consistency"

    # Convert images to bytes
    base_bytes_list = [_pil_to_bytes(img) for img in base_images]
    target_bytes_list = [_pil_to_bytes(img) for img in target_images]

    # Prepare request payload
    data = {
        "base_images": base_bytes_list,
        "target_images": target_bytes_list,
        "metadatas": metadatas
    }

    # Send request
    session = _create_session()
    try:
        response = session.post(
            endpoint,
            data=pickle.dumps(data),
            timeout=timeout
        )
        response.raise_for_status()

        # Parse response
        result = pickle.loads(response.content)

        if "error" in result:
            raise RuntimeError(f"Server error: {result['error']}")

        return result["object_results"], result["background_results"]

    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(
            f"Cannot connect to EDiVal server at {url}. "
            f"Please ensure the server is running: "
            f"python reward_server/edival_reward_server.py"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Error calling EDiVal server: {e}") from e


def edival_if_score_client(device=None, server_url: Optional[str] = None, num_votes: Optional[int] = None):

    url = server_url or SERVER_URL
    _num_votes = num_votes if num_votes is not None else int(os.getenv("EDIVAL_IF_NUM_VOTES", "1"))

    def _safe_convert_images(imgs):
        if isinstance(imgs, torch.Tensor):
            imgs = (imgs * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            imgs = imgs.transpose(0, 2, 3, 1)
            imgs = [Image.fromarray(img) for img in imgs]
        elif isinstance(imgs, list) and len(imgs) > 0 and isinstance(imgs[0], np.ndarray):
            imgs = [Image.fromarray(img) for img in imgs]
        return imgs

    def _fn(ref_images, images, prompts, metadatas):
        images = _safe_convert_images(images)
        ref_images = _safe_convert_images(ref_images)

        all_scores = []
        all_reasons = []
        num_batches = (len(images) + BATCH_SIZE - 1) // BATCH_SIZE

        for i in range(num_batches):
            start = i * BATCH_SIZE
            end = min((i + 1) * BATCH_SIZE, len(images))
            batch_ref = ref_images[start:end]
            batch_img = images[start:end]
            batch_prompts = prompts[start:end]
            batch_meta = metadatas[start:end]

            try:
                if _num_votes == 1:
                    if_scores, reasons = evaluate_via_server(
                        batch_ref, batch_img, batch_prompts, batch_meta, url
                    )
                else:
                    # Call _num_votes times in parallel; average scores to stabilise via vLLM TP non-determinism
                    with _ClientThreadPool(max_workers=_num_votes) as pool:
                        futures = [
                            pool.submit(evaluate_via_server, batch_ref, batch_img, batch_prompts, batch_meta, url)
                            for _ in range(_num_votes)
                        ]
                        vote_results = [f.result() for f in futures]

                    # vote_results: list of (scores_list, reasons_list); average scores across votes
                    vote_scores_matrix = [res[0] for res in vote_results]  # [num_votes, batch_size]
                    if_scores = [
                        float(np.mean([vote_scores_matrix[v][j] for v in range(_num_votes)]))
                        for j in range(len(batch_img))
                    ]
                    reasons = vote_results[0][1]  # use the first vote's reasons as representative
            except Exception as e:
                print(f'[IF WARNING] edival_if: error (batch {i+1}/{num_batches}): {e}')
                print(f'[IF WARNING] Falling back to 0.5 for this batch to keep training alive')
                if_scores = [0.5] * len(batch_img)
                reasons = [f'fallback: {e}'] * len(batch_img)

            all_scores.extend(if_scores)
            all_reasons.extend(reasons)

        return all_scores, {"reasons": all_reasons}

    return _fn


def edival_cc_score_client(device=None, server_url: Optional[str] = None):

    url = server_url or SERVER_URL

    def _safe_convert_images(imgs):
        if isinstance(imgs, torch.Tensor):
            imgs = (imgs * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            imgs = imgs.transpose(0, 2, 3, 1)
            imgs = [Image.fromarray(img) for img in imgs]
        elif isinstance(imgs, list) and len(imgs) > 0 and isinstance(imgs[0], np.ndarray):
            imgs = [Image.fromarray(img) for img in imgs]
        return imgs

    def _safe_cc(obj_val, bg_val, use_bg=True):
        if not use_bg:
            return obj_val if obj_val is not None else 0.5
        vals = [v for v in [obj_val, bg_val] if v is not None]
        return sum(vals) / len(vals) if vals else 0.5

    def _fn(base_images, images, metadatas):
        images = _safe_convert_images(images)
        base_images = _safe_convert_images(base_images)

        all_scores = []
        all_object_results = []
        all_background_results = []
        num_batches = (len(images) + BATCH_SIZE - 1) // BATCH_SIZE

        for i in range(num_batches):
            start = i * BATCH_SIZE
            end = min((i + 1) * BATCH_SIZE, len(images))
            batch_base = base_images[start:end]
            batch_img = images[start:end]
            batch_metadatas = metadatas[start:end]

            consistency_metadatas = [
                {
                    "unchanged_objects": meta.get("unchanged_objects", []),
                    "all_objects": meta.get("all_objects", [])
                }
                for meta in batch_metadatas
            ]

            try:
                object_results, background_results = evaluate_consistency_via_server(
                    batch_base, batch_img, consistency_metadatas, url
                )
                cc_scores = [
                    _safe_cc(
                        object_results[j]['object_dinov3_consistency_mean'],
                        background_results[j]['bg_dinov3_masked_similarity'],
                        use_bg=batch_metadatas[j].get('bg_consistency', True)
                    )
                    for j in range(len(object_results))
                ]
            except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as e:
                print(f'[CC WARNING] edival_cc: timeout (batch {i+1}/{num_batches}): {e}')
                cc_scores = [0.5] * len(batch_img)
                object_results = [{'object_dinov3_consistency_mean': None}] * len(batch_img)
                background_results = [{'bg_dinov3_masked_similarity': None}] * len(batch_img)
            except Exception as e:
                print(f'[CC WARNING] edival_cc: error (batch {i+1}/{num_batches}): {e}')
                cc_scores = [0.5] * len(batch_img)
                object_results = [{'object_dinov3_consistency_mean': None}] * len(batch_img)
                background_results = [{'bg_dinov3_masked_similarity': None}] * len(batch_img)

            all_scores.extend(cc_scores)
            all_object_results.extend(object_results)
            all_background_results.extend(background_results)

        return all_scores, {
            "object_results": all_object_results,
            "background_results": all_background_results
        }

    return _fn


def edival_score_client(device=None, server_url: Optional[str] = None):

    url = server_url or SERVER_URL

    def _fn(ref_images, images, prompts, metadatas, base_images=None):
        # Convert tensors to PIL images
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(img) for img in images]

        if isinstance(ref_images, torch.Tensor):
            ref_images = (ref_images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            ref_images = ref_images.transpose(0, 2, 3, 1)
            ref_images = [Image.fromarray(img) for img in ref_images]
        elif isinstance(ref_images, list) and isinstance(ref_images[0], np.ndarray):
            ref_images = [Image.fromarray(img) for img in ref_images]

        # Convert base_images (used for consistency evaluation)
        if base_images is not None:
            if isinstance(base_images, torch.Tensor):
                base_images = (base_images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                base_images = base_images.transpose(0, 2, 3, 1)
                base_images = [Image.fromarray(img) for img in base_images]
            elif isinstance(base_images, list) and isinstance(base_images[0], np.ndarray):
                base_images = [Image.fromarray(img) for img in base_images]

        # Read COEF from env var (set by the training script via config.edival_if_coef).
        # COEF=1.0: IF only; COEF=0.0: CC only; 0<COEF<1.0: parallel IF+CC
        COEF = float(os.getenv("EDIVAL_IF_COEF", "1.0"))

        # Process in batches to avoid sending too much data in a single request
        all_scores = []
        all_reasons = []
        all_object_results = []
        all_background_results = []

        batch_size = BATCH_SIZE
        num_batches = (len(images) + batch_size - 1) // batch_size

        def _safe_cc(obj_val, bg_val):
            vals = [v for v in [obj_val, bg_val] if v is not None]
            return sum(vals) / len(vals) if vals else 0.5

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(images))

            batch_ref = ref_images[start:end]
            batch_img = images[start:end]
            batch_prompts = prompts[start:end]
            batch_metadatas = metadatas[start:end]

            assert base_images is not None
            batch_base = base_images[start:end]
            consistency_metadatas = [
                {
                    "unchanged_objects": meta.get("unchanged_objects", []),
                    "all_objects": meta.get("all_objects", [])
                }
                for meta in batch_metadatas
            ]

            if COEF == 1.0:
                # IF only — no thread pool needed
                try:
                    IF_scores, reasons = evaluate_via_server(
                        batch_ref, batch_img, batch_prompts, batch_metadatas, url
                    )
                except Exception as e:
                    print(f'[IF WARNING] evaluate_via_server error (batch {i+1}/{num_batches}): {e}')
                    print(f'[IF WARNING] Falling back to 0.5 to keep training alive')
                    IF_scores = [0.5] * len(batch_img)
                    reasons = [f'fallback: {e}'] * len(batch_img)
                CC_scores = [0.5] * len(batch_img)
                object_results = [{'object_dinov3_consistency_mean': None} for _ in range(len(batch_img))]
                background_results = [{'bg_dinov3_masked_similarity': None} for _ in range(len(batch_img))]
            elif COEF == 0.0:
                # CC only — no thread pool needed
                try:
                    object_results, background_results = evaluate_consistency_via_server(
                        batch_base, batch_img, consistency_metadatas, url
                    )
                    CC_scores = [
                        _safe_cc(
                            object_results[j]['object_dinov3_consistency_mean'],
                            background_results[j]['bg_dinov3_masked_similarity']
                        )
                        for j in range(len(object_results))
                    ]
                except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as e:
                    print(f'[CC WARNING] evaluate_consistency_via_server timeout (batch {i+1}/{num_batches}): {e}')
                    print(f'[CC WARNING] Falling back to 0.5 to keep training alive')
                    CC_scores = [0.5] * len(batch_img)
                    object_results = [{'object_dinov3_consistency_mean': None} for _ in range(len(batch_img))]
                    background_results = [{'bg_dinov3_masked_similarity': None} for _ in range(len(batch_img))]
                except Exception as e:
                    print(f'[CC WARNING] evaluate_consistency_via_server error (batch {i+1}/{num_batches}): {e}')
                    print(f'[CC WARNING] Falling back to 0.5 to keep training alive')
                    CC_scores = [0.5] * len(batch_img)
                    object_results = [{'object_dinov3_consistency_mean': None} for _ in range(len(batch_img))]
                    background_results = [{'bg_dinov3_masked_similarity': None} for _ in range(len(batch_img))]
                IF_scores = [0.5] * len(batch_img)
                reasons = ['skipped (COEF=0.0)'] * len(batch_img)
            else:
                # Send IF and CC requests in parallel (wall time = max(IF, CC) instead of IF+CC)
                with _ClientThreadPool(max_workers=2) as pool:
                    if_future = pool.submit(
                        evaluate_via_server,
                        batch_ref, batch_img, batch_prompts, batch_metadatas, url
                    )
                    cc_future = pool.submit(
                        evaluate_consistency_via_server,
                        batch_base, batch_img, consistency_metadatas, url
                    )

                    try:
                        IF_scores, reasons = if_future.result()
                    except Exception as e:
                        print(f'[IF WARNING] evaluate_via_server error (batch {i+1}/{num_batches}): {e}')
                        print(f'[IF WARNING] Falling back to 0.5 to keep training alive')
                        IF_scores = [0.5] * len(batch_img)
                        reasons = [f'fallback: {e}'] * len(batch_img)

                    # Retrieve CC result (with timeout protection)
                    try:
                        object_results, background_results = cc_future.result()
                        CC_scores = [
                            _safe_cc(
                                object_results[j]['object_dinov3_consistency_mean'],
                                background_results[j]['bg_dinov3_masked_similarity']
                            )
                            for j in range(len(object_results))
                        ]
                    except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as e:
                        print(f'[CC WARNING] evaluate_consistency_via_server timeout (batch {i+1}/{num_batches}): {e}')
                        print(f'[CC WARNING] Falling back to 0.5 to keep training alive')
                        CC_scores = [0.5] * len(batch_img)
                        object_results = [{'object_dinov3_consistency_mean': None} for _ in range(len(batch_img))]
                        background_results = [{'bg_dinov3_masked_similarity': None} for _ in range(len(batch_img))]
                    except Exception as e:
                        print(f'[CC WARNING] evaluate_consistency_via_server error (batch {i+1}/{num_batches}): {e}')
                        print(f'[CC WARNING] Falling back to 0.5 to keep training alive')
                        CC_scores = [0.5] * len(batch_img)
                        object_results = [{'object_dinov3_consistency_mean': None} for _ in range(len(batch_img))]
                        background_results = [{'bg_dinov3_masked_similarity': None} for _ in range(len(batch_img))]


            # Combine IF and CC scores
            final_scores = list(COEF * np.array(IF_scores) + (1-COEF) * np.array(CC_scores))

            all_scores.extend(final_scores)
            all_reasons.extend(reasons)
            all_object_results.extend(object_results)
            all_background_results.extend(background_results)

        return all_scores, {
            "reasons": all_reasons,
            "object_results": all_object_results,
            "background_results": all_background_results
        }

    return _fn


def check_server_health(server_url: Optional[str] = None) -> Dict:
    """
    Check server health status.

    Returns:
        dict: Server status information.
    """
    url = server_url or SERVER_URL
    try:
        response = requests.get(f"{url}/health", timeout=5)
        response.raise_for_status()
        return pickle.loads(response.content)
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def _pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert a PIL Image to a base64 data URL."""
    buffered = BytesIO()
    image.save(buffered, format=format)
    encoded = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{encoded}"


def _call_vlm_ga(images: List[Image.Image], prompt: str, vlm_url: str, model_path: str,
                 temperature: float = 0.6) -> str:
    """Call an OpenAI-compatible VLM API with multiple images and a text prompt; return the response text."""
    content = []
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": _pil_to_base64(img)}})
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": model_path,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": 1024,
    }

    session = _create_session()
    try:
        response = session.post(
            f"{vlm_url}/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        return "no"
    except Exception as e:
        print(f"[GA VLM Error] {e}")
        return "no"


def _parse_ga_final_answer(response: str) -> float:
    """Extract <answer_final> yes/no </answer_final> from a VLM response; return 1.0 or 0.0."""
    match = re.search(r'<answer_final>\s*(yes|no)\s*</answer_final>', response, re.IGNORECASE)
    if match:
        return 1.0 if match.group(1).lower() == 'yes' else 0.0
    return 0.0


def _to_pil(img) -> Image.Image:
    """Uniformly convert a tensor, numpy array, or PIL Image to a PIL Image."""
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, torch.Tensor):
        arr = (img.cpu().float().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    if isinstance(img, np.ndarray):
        return Image.fromarray(img.astype(np.uint8))
    raise TypeError(f"Unsupported image type: {type(img)}")


def edival_ga_score_client(device=None, server_url=None, num_votes: Optional[int] = None,
                           temperature: Optional[float] = None):

    vlm_url = server_url or GA_VLM_URL
    model_path = GA_MODEL_PATH
    _num_votes = num_votes if num_votes is not None else int(os.getenv("EDIVAL_GA_NUM_VOTES", "1"))
    _temperature = temperature if temperature is not None else float(os.getenv("EDIVAL_GA_TEMPERATURE", "0.6"))

    def _process_single(imgs: List, instrs: List[str]) -> Tuple[float, str]:
        pil_imgs = [_to_pil(img) for img in imgs]
        instructions_formatted = "\n".join(f"第{i+1}轮：{instr}" for i, instr in enumerate(instrs))
        prompt = MULTITURN_VLM_PROMPT.format(instructions_formatted=instructions_formatted)

        if _num_votes == 1:
            response = _call_vlm_ga(pil_imgs, prompt, vlm_url, model_path, temperature=_temperature)
            score = _parse_ga_final_answer(response)
            return score, response
        else:
            # Call _num_votes times in parallel; average scores to stabilise via vLLM TP non-determinism
            _t0 = time.time()
            with _ClientThreadPool(max_workers=_num_votes) as pool:
                futures = [
                    pool.submit(_call_vlm_ga, pil_imgs, prompt, vlm_url, model_path, _temperature)
                    for _ in range(_num_votes)
                ]
                raw_responses = [f.result() for f in futures]
            _t1 = time.time()

            vote_responses = []
            vote_scores = []
            for v, resp in enumerate(raw_responses):
                s = _parse_ga_final_answer(resp)
                vote_responses.append(resp)
                vote_scores.append(s)
            score = float(np.mean(vote_scores))
            return score, vote_responses[0]  # use the first vote's response as representative

    def _fn(images_history: List[List], prompts_history: List[List[str]]) -> Tuple[List[float], Dict]:
        all_scores = []
        all_responses = []

        with _ClientThreadPool(max_workers=min(8, len(images_history))) as pool:
            futures_list = [
                pool.submit(_process_single, imgs, instrs)
                for imgs, instrs in zip(images_history, prompts_history)
            ]
            for idx, future in enumerate(futures_list):
                try:
                    score, response = future.result()
                except Exception as e:
                    print(f'[GA WARNING] edival_ga: error on sample {idx}: {e}')
                    print(f'[GA WARNING] Falling back to 0.5 for this sample to keep training alive')
                    score, response = 0.5, f'fallback: {e}'
                all_scores.append(score)
                all_responses.append(response)

        return all_scores, {"responses": all_responses}

    return _fn


def edival_cu_score_client(device=None, server_url=None, num_votes: Optional[int] = None,
                           temperature: Optional[float] = None):

    vlm_url = server_url or GA_VLM_URL
    model_path = GA_MODEL_PATH
    _num_votes = num_votes if num_votes is not None else int(os.getenv("EDIVAL_GA_NUM_VOTES", "1"))
    _temperature = temperature if temperature is not None else float(os.getenv("EDIVAL_GA_TEMPERATURE", "0.6"))

    def _process_single(imgs: List, instrs: List[str], fmt_instrs: List[str]) -> Tuple[float, str]:
        pil_imgs = [_to_pil(img) for img in imgs]
        instructions_formatted = "\n".join(
            f"第{i+1}轮：\n  - 模型接收的指令：{instr}\n  - 显式参照指令：{fmt_instr}"
            for i, (instr, fmt_instr) in enumerate(zip(instrs, fmt_instrs))
        )
        prompt = MULTITURN_CU_VLM_PROMPT.format(instructions_formatted=instructions_formatted)

        if _num_votes == 1:
            response = _call_vlm_ga(pil_imgs, prompt, vlm_url, model_path, temperature=_temperature)
            score = _parse_ga_final_answer(response)
            return score, response
        else:
            _t0 = time.time()
            with _ClientThreadPool(max_workers=_num_votes) as pool:
                futures = [
                    pool.submit(_call_vlm_ga, pil_imgs, prompt, vlm_url, model_path, _temperature)
                    for _ in range(_num_votes)
                ]
                raw_responses = [f.result() for f in futures]
            _t1 = time.time()

            vote_responses = []
            vote_scores = []
            for v, resp in enumerate(raw_responses):
                s = _parse_ga_final_answer(resp)
                vote_responses.append(resp)
                vote_scores.append(s)
            score = float(np.mean(vote_scores))
            return score, vote_responses[0]

    def _fn(images_history: List[List], prompts_history: List[List[str]],
            format_instructions_history: List[List[str]]) -> Tuple[List[float], Dict]:
        all_scores = []
        all_responses = []

        with _ClientThreadPool(max_workers=min(8, len(images_history))) as pool:
            futures_list = [
                pool.submit(_process_single, imgs, instrs, fmt_instrs)
                for imgs, instrs, fmt_instrs in zip(images_history, prompts_history, format_instructions_history)
            ]
            for idx, future in enumerate(futures_list):
                try:
                    score, response = future.result()
                except Exception as e:
                    score, response = 0.5, f'fallback: {e}'
                all_scores.append(score)
                all_responses.append(response)

        return all_scores, {"responses": all_responses}

    return _fn


# Backward compatibility alias
edival_score = edival_score_client


