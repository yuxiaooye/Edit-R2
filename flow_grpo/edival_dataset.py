import os
import json
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class EDiValPromptImageDataset(Dataset):

    
    def __init__(self, dataset, resolution=512, split="train", use_random_images=False):
        self.dataset = dataset
        self.resolution = resolution
        self.use_random_images = use_random_images
        self.file_path = os.path.join(dataset, f"{split}_metadata.jsonl")
        
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            
        self.prompts = []
        for item in self.metadatas:
            prompt = item.get("instruction", item.get("prompt", ""))
            
            # --- add for multi-turn editing
            if isinstance(prompt, list): # 
                prompt = " | ".join(prompt)   
            # --- add for multi-turn editing 

            self.prompts.append(prompt)
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        item = {
            "prompt": self.prompts[idx],
            "metadata": self.metadatas[idx]
        }
        
        image_path = self.metadatas[idx].get("image", "")
        
        item["prompt_with_image_path"] = f"{self.prompts[idx]}_{image_path}"
        
        if self.use_random_images:
            np.random.seed(idx + hash(self.prompts[idx]) % 10000)
            random_array = np.random.randint(
                0, 255, 
                (self.resolution, self.resolution, 3), 
                dtype=np.uint8
            )
            image = Image.fromarray(random_array)
        else:
            full_image_path = os.path.join(self.dataset, image_path)
            image = Image.open(full_image_path).convert("RGB")
            
            # w, h = image.size
            # if w != h:
            #     min_dim = min(w, h)
            #     left = (w - min_dim) // 2
            #     top = (h - min_dim) // 2
            #     right = left + min_dim
            #     bottom = top + min_dim
            #     image = image.crop((left, top, right, bottom))
            
            
            image = image.resize(
                (self.resolution, self.resolution), 
                Image.Resampling.LANCZOS
            )
        
        item["image"] = image
        return item
    
    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        images = [example["image"] for example in examples]
        prompt_with_image_paths = [
            example["prompt_with_image_path"] for example in examples
        ]
        return prompts, metadatas, images, prompt_with_image_paths




def create_metadata_example(
    image_path: str,
    instruction: str,
    task_type: str,
    formatted_instruction: str,
    prompt: str = None
) -> dict:
    metadata = {
        "image": image_path,
        "instruction": instruction,
        "task_type": task_type,
        "formatted_instruction": formatted_instruction,
    }
    
    if prompt is not None and prompt != instruction:
        metadata["prompt"] = prompt
    
    return metadata


