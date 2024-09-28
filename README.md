<div align="center">
<h1>Magnet: We Never Know How Text-to-Image Diffusion Models Work, Until We Learn How Vision-Language Models Function</h1>


[Chenyi Zhuang](https://chenyi-zhuang.github.io/), Ying Hu, Pan Gao

[I2ML](https://i2-multimedia-lab.github.io/), Nanjing University of Aeronautics and Astronautics

[Paper]()

<p><B>We propose Magnet, a training-free approach that improves attribute binding by manipulating object embeddings, enhancing disentanglement within the textual space.</B></p>
<img src="./figures/magnet_workflow.jpg" width="800px">
</div>


### 🌟 Key Features
1. In-depth analysis and exploration of the CLIP text encoder, highlighting the context issue of padding embeddings;
2. Improve text alignment by applying positive and negative binding vectors on object embeddings, with negligible cost.
3. Plug-and-play to various T2I models and controlling methods, e.g., ControlNet.

### ⚙️ Setup and Usage
```bash
conda create --name magnet python=3.11
conda activate magnet

# Install requirements
pip install -r requirements.txt
```

If you are curious about how different types of text embedding influence generation, we recommend running (1) ``visualize_attribute_bias.ipynb`` to explore the attribute bias on different objects, (2) ``emb_swap_cases.py`` to reproduce the swapping experiment.

Download the pre-trained [SD V1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4),  [SD V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) (unfortunately now 404), [SD V2](https://huggingface.co/stabilityai/stable-diffusion-2-base), [SD V2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base), or [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).

```python
# Run magnet on SD V1.4
python run.py --sd_path path-to-stable-diffusion-v1-4 --magnet_path bank/candidates_1_4.pt --N 2 --run_sd

# Run magnet on SDXL
python run.py --sd_path path-to-stable-diffusion-xl --magnet_path bank/candidates_sdxl.pt --N 2 --run_sd

# Remove the "run_sd" argument if you don't want the standard model run
```

You can also try [ControlNet](https://huggingface.co/lllyasviel/sd-controlnet-depth) conditioned on Depth estimation [DPT-Large](https://huggingface.co/Intel/dpt-large).

```python
# Run magnet with ControlNet
python run_with_controlnet.py --sd_path path-to-stable-diffusion-v1-5 --magnet_path bank/candidates_1_5.pt --N 2 --controlnet_path path-to-sd-controlnet-depth --dpt_path path-to-dpt-large --run_sd
```

We also provide ```run_vanilla_pipeline.py``` to use magnet via the ```prompt_embeds``` argument in the standard ```StableDiffusionPipeline```. 


Demos of cross-attention visualization are in ``visualize_attention.ipynb``.

<p><B>Feel free to explore Magnet and leave any questions in this repo!</B></p>

### 😺 Examples
Compare to state-of-the-art approaches:

<img src="./figures/qualitative.jpg" width="800px">

Integrate Magnet into other T2I pipelines and T2I controlling modules:

<img src="./figures/qualitative_extention.jpg" width="800px">

### 😿 Limitations
Magnet's performance is largely dependent on the pre-trained T2I model. It may not provide meaningful modifications due to the limited power of text-based manipulation alone. You can manually adjust the prompt, seed, or hyperparameters, and combine other techniques to get a better result if you are not satisfied with the output.

### 🌊 Acknowledgements
Most prompts are based on datasets obtained from [Structure Diffusion](https://github.com/weixi-feng/Structured-Diffusion-Guidance). We also refer to some excellent demos and repos, including [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt) and [ControlNet](https://github.com/lllyasviel/ControlNet).

## TODO
- [x] Release the source code and model.
- [x] Extend to more T2I models.
- [x] Extend to controlling approaches.


