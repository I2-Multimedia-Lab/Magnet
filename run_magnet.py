import pprint
import os
from typing import List
import torch
from PIL import Image
from pipeline_sd import MagnetSDPipeline
from pipeline_sdxl import MagnetSDXLPipeline
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from utils.magnet_utils import *
from utils.ptp_utils import *
from tqdm import tqdm, trange
from pytorch_lightning import seed_everything
from transformers import pipeline
from torch.nn import functional as F
import time


model_lists = [
    "path-to-stable-diffusion-v1-4",
    "path-to-stable-diffusion-v1-5",
    "path-to-stable-diffusion-2-base",
    "path-to-stable-diffusion-2-1-base",
    "path-to-stable-diffusion-xl-base-1.0"
]
offline_files = [
    "./bank/candidates_1_4.pt",
    "./bank/candidates_1_5.pt",
    "./bank/candidates_2.pt",
    "./bank/candidates_2_1.pt",
    "./bank/candidates_sdxl.pt"
]


def load_bert(model_path=None):
    if model_path is None:
        model_path = 'bert-base-uncased'
    return pipeline('fill-mask', model=model_path)


@torch.no_grad()
def main(): 

    version_idx = 0

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if "xl" in model_lists[version_idx]:
        pipe = MagnetSDXLPipeline.from_pretrained(model_lists[version_idx]).to(device)
    else:
        pipe = MagnetSDPipeline.from_pretrained(model_lists[version_idx]).to(device)

    pipe.prepare_candidates(offline_file=offline_files[version_idx])

    SD_2 = True if "2" in model_lists[version_idx] else False

    output_path = "outputs"
    os.makedirs(output_path, exist_ok=True)

    test_prompts = [
        "a lone, green fire hydrant sits in red grass",
        "some blue bananas with little yellow stickers on them",
        "a bowl of broccoli and red rice with a white sauce"
    ]

    sample_per_prompt = 5
    
    for bid, prompt in enumerate(test_prompts):

        for METHOD in ["sd", "magnet"]:
            if METHOD == "magnet":
                try:
                    with torch.no_grad():
                        pipe.get_magnet_direction(prompt, alpha_lambda=0.6, neighbor="feature", K=5, sd_2=SD_2)
                except:
                    print(f"Fail to apply Magnet at prompt: {prompt}")
                    pipe.magnet_embeddings = None
            else:
                pipe.magnet_embeddings = None
            
            for i in range(sample_per_prompt):
                cur_seed = 123 + i * 55 + bid * 77
                seed_everything(int(cur_seed))
                
                outputs = pipe(
                    prompt=prompt,
                    guidance_scale=7.5,
                    guidance_rescale=0.,
                    num_inference_steps=50,
                    num_images_per_prompt=1
                ).images
                
                outputs[0].save(os.path.join(output_path, f'{bid+1}_seed{cur_seed}_{METHOD}_{prompt}.png'))
                    
                
if __name__ == '__main__':
    main()
