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
import argparse


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
def main(opt): 

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if "xl" in opt.sd_path:
        pipe = MagnetSDXLPipeline.from_pretrained(opt.sd_path, torch_dtype=opt.dtype).to(device)
    else:
        pipe = MagnetSDPipeline.from_pretrained(opt.sd_path, torch_dtype=opt.dtype).to(device)

    pipe.prepare_candidates(offline_file=opt.magnet_path)

    output_path = "outputs"
    os.makedirs(output_path, exist_ok=True)

    default_prompts = [
        "a lone, green fire hydrant sits in red grass",
        "some blue bananas with little yellow stickers on them",
        "a bowl of broccoli and red rice with a white sauce"
    ]
    test_prompts = default_prompts if opt.prompts is None else opt.prompts

    METHOD_LIST = ["sd", "magnet"] if opt.run_sd else ["magnet"]
    
    for bid, prompt in enumerate(test_prompts):

        for METHOD in METHOD_LIST:
            if METHOD == "magnet":
                try:
                    with torch.no_grad():
                        pipe.get_magnet_direction(prompt, alpha_lambda=opt.L, K=opt.K, neighbor="feature")
                except:
                    print(f"Fail to apply Magnet at prompt: {prompt}")
                    break
            else:
                pipe.magnet_embeddings = None
            
            cur_seed = 14273 + bid * 55
            seed_everything(int(cur_seed))
            
            outputs = pipe(
                prompt=prompt,
                guidance_scale=opt.cfg_scale,
                num_inference_steps=opt.ddim_steps,
                num_images_per_prompt=opt.N
            ).images

            [outputs[i].save(os.path.join(output_path, f'{prompt}_seed{cur_seed}_{i}_{METHOD}.png')) for i in range(opt.N)]
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sd_path', type=str, required=True,
                        help='Path to pre-trained Stable Diffusion.')
    parser.add_argument('--magnet_path', type=str, 
                        help='Path to the local file of the candidate embedding to save time.'
                        )
    parser.add_argument('--prompts', nargs='+', default=None,
                        help='Prompt list to generate a batch of images.')
    
    # Magnet settings
    parser.add_argument('--K', type=int, default=5,
                        help='Hyperparameter of Magnet for the number of neighbor objects.')
    parser.add_argument('--L', type=float, default=0.6,
                        help='Hyperparameter of Magnet for adaptive strength.')
    
    # SD settings
    parser.add_argument('--dtype', type=str, default=torch.float16, 
                        help='Override the default `torch.dtype` and load the model with another dtype.')
    parser.add_argument('--cfg_scale', type=float, default=7.5)
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--N', type=int, default=1,
                        help='Number of image for each prompt.')
    parser.add_argument('--run_sd', action='store_true')
    
    args = parser.parse_args()
    
    main(args)
