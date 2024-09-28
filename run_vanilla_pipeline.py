import pprint
import os
from typing import List
import torch
from PIL import Image
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from utils.ptp_utils import *
from utils.magnet import *
from pytorch_lightning import seed_everything
import time
import argparse


@torch.no_grad()
def main(opt): 

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    pipe = StableDiffusionPipeline.from_pretrained(opt.sd_path).to(device)
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    parser = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', download_method=None)
    candidates, candidate_embs = prepare_candidates(offline_file=opt.magnet_path)
    candidates = candidates.to(device)
    candidate_embs = candidate_embs.to(device)

    SD_2 = True if "2" in opt.sd_path else False

    output_path = "outputs"
    os.makedirs(output_path, exist_ok=True)

    default_prompts = [
        "a lone, green fire hydrant sits in red grass",
        "some blue bananas with little yellow stickers on them",
        "a bowl of broccoli and red rice with a white sauce"
    ]
    test_prompts = default_prompts if opt.prompts is None else opt.prompts

    num_images_per_prompt = opt.N

    METHOD_LIST = ["sd", "magnet"] if opt.run_sd else ["magnet"]
    
    for bid, prompt in enumerate(test_prompts):

        for METHOD in METHOD_LIST:
            if METHOD == "magnet":
                try:
                    with torch.no_grad():
                        magnet_embeds = get_magnet_direction(
                            parser,
                            tokenizer, 
                            text_encoder,
                            prompt, 
                            candidates,
                            candidate_embs,
                            alpha_lambda=opt.L, neighbor="feature", K=opt.K, sd_2=SD_2
                        )
                except:
                    print(f"Fail to apply Magnet at prompt: {prompt}")
                    break
            else:
                magnet_embeds = None
            
            cur_seed = 14273 + bid * 55
            seed_everything(int(cur_seed))
            
            if magnet_embeds is None:
                outputs = pipe(
                    prompt=prompt,
                    guidance_scale=opt.cfg_scale,
                    guidance_rescale=0.,
                    num_inference_steps=opt.ddim_steps,
                    num_images_per_prompt=num_images_per_prompt
                ).images
            else:
                outputs = pipe(
                    guidance_scale=opt.cfg_scale,
                    guidance_rescale=0.,
                    prompt_embeds=magnet_embeds,
                    num_inference_steps=opt.ddim_steps,
                    num_images_per_prompt=num_images_per_prompt
                ).images
            
            [outputs[i].save(os.path.join(output_path, f'{prompt}_seed{cur_seed}_{i}_{METHOD}.png')) for i in range(num_images_per_prompt)]
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sd_path', type=str, required=True,
                        help='Path to pre-trained Stable Diffusion.')
    parser.add_argument('--magnet_path', type=str, 
                        help='Path to the local file of the candidate embedding to save time.'
                        )
    parser.add_argument('--prompts', type=list, default=None,
                        help='Prompt list to generate a batch of images.')
    
    # Magnet settings
    parser.add_argument('--K', type=int, default=5,
                        help='Hyperparameter of Magnet for the number of neighbor objects.')
    parser.add_argument('--L', type=float, default=0.6,
                        help='Hyperparameter of Magnet for adaptive strength.')
    
    # SD settings
    parser.add_argument('--cfg_scale', type=float, default=7.5)
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--N', type=int, default=1,
                        help='Number of image for each prompt.')
    parser.add_argument('--run_sd', action='store_true')
    
    args = parser.parse_args()
    
    main(args)
