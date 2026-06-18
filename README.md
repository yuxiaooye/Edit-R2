<h1 align="center"> Edit-R2: Multi-Turn Image Editing via Flow-GRPO </h1>

<div align="center">
  <a href="https://arxiv.org/abs/2505.05470"><img src="https://img.shields.io/badge/ArXiv-Edit--R2-red?logo=arxiv" /></a> &nbsp;
</div>

<br>




Edit-R2 formulates **multi-turn in-context image editing** as a session-level RL problem, training [BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) with online RL. It features:

- **In-Context Chain-of-Thought (IC-CoT)** — BAGEL's own vision-language pathway reconstructs the operative session intent from the dialogue history, consolidating scattered historical constraints into a compact reasoning trace before sampling
- **Unified multi-turn RL** — IC-CoT generation and visual generation are jointly optimized, coupling the discrete text space of session-intent reconstruction with the continuous latent space of the flow-matching generator
- **Trajectory filtering** — prevents corrupted rollouts from dominating training and contaminating later training steps, ensuring stable session-level optimization
- **MICE-Bench reward suite** — three complementary reward signals: Instruction Following (IF), Content Consistency (CC, following EDiVal), and the newly introduced Global Awareness (GA), measuring compliance with accumulated session-level constraints


<img width="2082" height="1060" alt="teaser" src="https://github.com/user-attachments/assets/ecdc162e-9fd6-4739-a85d-6e4a613a0a62" />

---

## 🔧 Installation

We use separate env for training and reward model deployment:
```bash
# for training
git clone <repo_url> Edit-R2
cd Edit-R2
conda create -n edit-r2 python=3.10 -y
conda activate edit-r2
pip install -e .
pip install flash-attn==2.7.4.post1 --no-build-isolation # required for BAGEL

# for reward model deployment
conda create -n rewards python=3.11 -y
conda activate rewards
pip install -r requirements_reward.txt
```

---

## Model Preparation

Download the BAGEL-7B-MoT as basemodel; download Qwen2.5-VL-32B-Instruct and DINOv3 for reward server:
```
huggingface-cli download ByteDance-Seed/BAGEL-7B-MoT --local-dir <your path>
huggingface-cli download Qwen/Qwen2.5-VL-32B-Instruct --local-dir <your path>
huggingface-cli download timm/vit_large_patch16_dinov3.lvd1689m --local-dir <your path>
```

Download GroundingDINO for reward server:
```
cd rewards
bash install_grounding.sh
```
---

## Reward Server Setup

Edit-R2 training relies on remote reward servers. Start the reward server **on every triaining node** (provides IF, CC and GA rewards):

```bash
# deploy VLM and GroundingDINO
export VLLM_MODEL_PATH=/path/to/Qwen2.5-VL-32B-Instruct
export GROUNDING_CONFIG_PATH=/path/to/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
export GROUNDING_WEIGHT_PATH=/path/to/GroundingDINO/weights/groundingdino_swint_ogc.pth
cd rewards
conda activate rewards
bash start_servers.sh $VLLM_MODEL_PATH 0,1 & # deploy VLM on first 2 GPUs
bash start_grounding.sh 0,1,2,3 & # deploy GroundingDINO on first 4 GPUs

# deploy DINOv3 and start reward server
export DINOV3_PATH=/path/to/DINOV3
nohup python -u reward_server/edival_reward_server.py \
  --port 12342 \
  --vllm-url http://localhost:8000/v1 \
  --grounding-dino-url http://localhost:12343 \
  --vlm-temperature 0.6 \
  > edival_reward_server_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# health check
python rm_health_check.py
```
You are expected to see:
```
EdiVal-IF score: 1
Reason: Successfully added [dog] to the left of [tree]...
```


---

## Dataset Format

Datasets are expected in **EDiVal format**: a directory containing `train_metadata.jsonl` and `test_metadata.jsonl` where each line is:

```json
{
  "image": "relative/path/to/ref_image.jpg",
  "instruction": "Change the color of the car to red",
  "task_type": "color_alter",
  "formatted_instruction": "Change the color of [car] to [red]",
  "unchanged_objects": ["tree", "road"],
  "all_objects": ["car", "tree", "road"]
}
```

<!-- | Field | Description |
|-------|-------------|
| `image` | Path to reference image, relative to the dataset root |
| `instruction` | Natural language instruction (may use pronouns like "it", "them") |
| `task_type` | Task category (e.g. `subject_add`, `color_alter`, `subject_replace`) |
| `formatted_instruction` | Bracketed version of instruction for CU reward evaluation |
| `unchanged_objects` | Objects that should remain unmodified (for CC reward) |
| `all_objects` | All objects in the scene (for CC reward) | -->

---

## 🚀 Quick Start

### Multi-node training (24 GPUs across 4 nodes)

Run the following command **on every triaining node**:

```bash
# Node 0 (master)
export MASTER_ADDR="<master_node_ip>"
export WANDB_API_KEY="<your_wandb_key>"
bash scripts/multi_node/bagel/train.sh 0
# Node 1
bash scripts/multi_node/bagel/train.sh 1
# Node 2
bash scripts/multi_node/bagel/train.sh 2
# Node 3
bash scripts/multi_node/bagel/train.sh 3
```


## Acknowledgement

This work builds on [Flow-GRPO](https://arxiv.org/abs/2505.05470), [BAGEL](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) and [EdiVal-Agent](https://github.com/TianyuCodings/EdiVal).
We thank the authors for their open-source contributions.

## Citation

```bibtex
@misc{ye2026editr2contextawarereinforcementlearning,
      title={Edit-R2: Context-Aware Reinforcement Learning for Multi-Turn Image Editing}, 
      author={Yuxiao Ye and Haoran He and Fangyuan Kong and Xintao Wang and Pengfei Wan and Kun Gai and Ling Pan},
      year={2026},
      eprint={2606.05950},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2606.05950}, 
}
```
