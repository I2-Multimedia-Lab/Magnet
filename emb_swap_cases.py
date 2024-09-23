import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionPipeline
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import json
import random
import argparse
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision.transforms as T
from pytorch_lightning import seed_everything
from utils import *


def check_emb_len(input, target_len):
    # input:[B, L, C], check length of L

    cur_len = input.shape[1]
    pad_len = target_len - cur_len

    if pad_len > 0:
        pad_v = input[:, -1:].repeat(1, pad_len, 1)
        output = torch.cat([input, pad_v], dim=1)
    elif pad_len < 0:
        output = input[:, :target_len, :]
    else:
        output = input

    return output


if __name__ == "__main__":
    model_key = "path-to-stable-diffusion-v1-4"
    NUM_DDIM_STEPS = 50
    GUIDANCE_SCALE = 7.5
    output_path = "./swap_exp"
    os.makedirs(output_path, exist_ok=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    pipe = StableDiffusionPipeline.from_pretrained(model_key).to("cuda")
    pipe.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
    pipe.scheduler.set_timesteps(NUM_DDIM_STEPS, device=device)
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder.eval()
    pipe.safety_checker = None

    colors = ["green", "red", "blue", "black", "white"]
    objects = ["car", "chair", "cat", "swan", "sheep", "apple", "banana", "broccoli"]

    per_instance_num = 20

    instance_num = 0
    for obj in objects:
        for attr in colors:

            prompt = " ".join([attr, obj])
            full_ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt",).input_ids.to(device)
            full_embs = text_encoder(full_ids)[0]

            prompt_wo_color = f"{obj}"
            wo_color_ids = tokenizer(prompt_wo_color, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt",).input_ids.to(device)
            wo_color_embs = text_encoder(wo_color_ids)[0]

            attr_ids = tokenizer(attr, padding=False, truncation=True, return_tensors="pt",).input_ids.to(device)
            attr_embs = text_encoder(attr_ids)[0][:, 1].unsqueeze(1).detach()
            obj_ids = tokenizer(obj, padding=False, truncation=True, return_tensors="pt",).input_ids.to(device)
            obj_embs = text_encoder(obj_ids)[0][:, 1].unsqueeze(1).detach()
    

            sot = full_embs[:, :1].clone().detach()

            word_w_context = full_embs[:, 1:3].clone().detach()
            word_wo_context = torch.cat([attr_embs, obj_embs], dim=1)

            eot_w_color = full_embs[:, 3:].clone().detach()
            eot_wo_color = wo_color_embs[:, 2:].clone().detach()

            le = tokenizer.model_max_length

            # case1 [v_sot, v_color, v_obj, v_eot, v_pad_1, ..., v_pad_L]
            case1 = check_emb_len(torch.cat([sot, word_w_context, eot_w_color], dim=1), le)
            case2 = check_emb_len(torch.cat([sot, word_wo_context, eot_w_color], dim=1), le)
            case3 = check_emb_len(torch.cat([sot, word_w_context, eot_wo_color], dim=1), le)
            case4 = check_emb_len(torch.cat([sot, word_wo_context, eot_wo_color], dim=1), le)
            main_cases = torch.cat([case1, case2, case3, case4], dim=0)
            print(main_cases.shape)  # [4, 77, 768]

            eot_w_color_A = full_embs[:, 3:27].clone().detach()
            eot_w_color_B = full_embs[:, 27:51].clone().detach()
            eot_w_color_C = full_embs[:, 51:-1].clone().detach()

            eot_wo_color_A = wo_color_embs[:, 2:27].clone().detach()
            eot_wo_color_B = wo_color_embs[:, 27:51].clone().detach()
            eot_wo_color_C = wo_color_embs[:, 51:-1].clone().detach()

            case_appendix_case1 = check_emb_len(torch.cat([sot, word_wo_context, eot_wo_color_A, eot_wo_color_B, eot_w_color_C], dim=1), le)
            case_appendix_case2 = check_emb_len(torch.cat([sot, word_wo_context, eot_wo_color_A, eot_w_color_B, eot_wo_color_C], dim=1), le)
            case_appendix_case3 = check_emb_len(torch.cat([sot, word_wo_context, eot_w_color_A, eot_wo_color_B, eot_wo_color_C], dim=1), le)
            appendix_cases = torch.cat([case_appendix_case1, case_appendix_case2, case_appendix_case3], dim=0)
            print(appendix_cases.shape)  # [3, 77, 768]

            instance_num += 1
            for pid in range(per_instance_num):
                seed = 133 + 55 * instance_num + pid * 77
                seed_everything(seed)

                cur_path = os.path.join(output_path, f"{attr}_{obj}")
                os.makedirs(cur_path, exist_ok=True)

                latents = torch.randn([1, 4, 64, 64])

                outputs = pipe(
                    prompt_embeds=main_cases,
                    latents=latents.repeat(4, 1, 1, 1),
                    guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=NUM_DDIM_STEPS,
                    num_images_per_prompt=1,
                ).images

                for i, output in enumerate(outputs):
                    output.save(os.path.join(cur_path, f'{pid}_case{i+1}_seed{seed}.png'))

                outputs = pipe(
                    prompt_embeds=appendix_cases,
                    latents=latents.repeat(3, 1, 1, 1),
                    guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=NUM_DDIM_STEPS,
                    num_images_per_prompt=1,
                ).images

                for i, output in enumerate(outputs):
                    output.save(os.path.join(cur_path, f'{pid}_appendix_case{i+1}_seed{seed}.png'))
