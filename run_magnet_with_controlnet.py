import pprint
import os
from typing import List
import torch
from PIL import Image
from pipeline_controlnet import MagnetSDControlNetPipeline
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from diffusers import ControlNetModel
from utils.magnet_utils import *
from utils.ptp_utils import *
from tqdm import tqdm, trange
from pytorch_lightning import seed_everything
from transformers import pipeline
from torch.nn import functional as F
import time
from transformers import DPTImageProcessor, DPTForDepthEstimation


def load_bert(model_path=None):
    if model_path is None:
        model_path = 'bert-base-uncased'
    return pipeline('fill-mask', model=model_path)


def get_depth(image_path, processor, model):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted).resize((512, 512))
    return depth


@torch.no_grad()
def main(): 

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    processor = DPTImageProcessor.from_pretrained("path-to-dpt-large")
    model = DPTForDepthEstimation.from_pretrained("path-to-dpt-large")

    depth_image = "./example_images/teddy_bear.jpg"
    input_depth = get_depth(depth_image, processor, model)

    controlnet = ControlNetModel.from_pretrained("path-to-sd-controlnet-depth")

    pipe = MagnetSDControlNetPipeline.from_pretrained(
        "path-to-stable-diffusion-v1-5",
        controlnet=controlnet
    ).to(device)

    pipe.prepare_candidates(offline_file="./bank/candidates_1_5.pt")

    output_path = "outputs"
    os.makedirs(output_path, exist_ok=True)

    test_prompts = [
        "a smiling teddy bear with white bow",
    ]
    
    for bid, prompt in enumerate(test_prompts):

        for METHOD in ["sd", "magnet"]:
            if METHOD == "magnet":
                try:
                    with torch.no_grad():
                        pipe.get_magnet_direction(prompt, alpha_lambda=0.6, neighbor="feature", K=5)
                except:
                    print(f"Fail to apply Magnet at prompt: {prompt}")
                    pipe.magnet_embeddings = None
            else:
                pipe.magnet_embeddings = None
            
            for i in range(5):
                cur_seed = 12 + i * 33 + bid * 123
                seed_everything(int(cur_seed))

                outputs = pipe(prompt, input_depth, num_inference_steps=50).images
                
                outputs[0].save(os.path.join(output_path, f'{bid+1}_seed{cur_seed}_{METHOD}_{prompt}.png'))
                    
                
if __name__ == '__main__':
    main()
