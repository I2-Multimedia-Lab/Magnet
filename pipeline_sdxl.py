import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers import StableDiffusionXLPipeline
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils import deprecate
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput

import numpy as np
from torch.nn import functional as F
from utils.magnet_utils import *


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps



class MagnetSDXLPipeline(StableDiffusionXLPipeline):

    def prepare_candidates(self, offline_file=None, save_path=None, obj_file="./bank/candidates.txt"):

        self.magnet_embeddings = None
        self.parser = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', download_method=None)

        with open(obj_file, "r") as f:
            candidates = f.read().splitlines()
        self.candidates = np.array(candidates)

        if offline_file is None:
            with torch.no_grad():
                self.candidate_embs = torch.cat([self.get_eot(w, -1) for w in candidates], dim=1)[0]
                self.candidate_embs = self.candidate_embs.to("cuda")
            if save_path is not None:
                torch.save(self.candidate_embs, save_path)
        else:
            self.candidate_embs = torch.load(offline_file).to("cuda")
        
        print("Finished loading candidate embeddings with shape:", self.candidate_embs.shape)

    def get_magnet_direction(
        self,
        prompt, 
        pairs=None,
        alphas=[1, 1],
        betas=[0.5, 0.5],
        K=5,
        alpha_lambda=0.6,
        use_neg = True,
        use_pos = True,
        neighbor = "feature",
        sd_2 = False
    ):
        assert len(self.candidates) == self.candidate_embs.shape[0]

        prompt = check_prompt(prompt)
        # print(prompt)
        text_inds = self.tokenizer.encode(prompt)
        self.eot_index = len(text_inds) - 1

        if pairs == None:
            pairs = get_pairs(prompt, self.parser)
            # print('Extracted Dependency : \n', pairs)

        prompt_embeds, eid = self.get_prompt_embeds_with_eid(prompt)

        # print(alphas, betas)
        N_pairs = len(pairs)

        for pid, pair in enumerate(pairs):
            # if pair["concept"] == pair["subject"]: continue

            # print(pair)
            cur_span = get_span(prompt, pair['concept'])
            cur_concept_index = get_word_inds(prompt, cur_span, tokenizer=self.tokenizer, text_inds=text_inds)
            
            concept_embeds, concept_eid = self.get_prompt_embeds_with_eid(pair['concept'])
            omega = F.cosine_similarity(concept_embeds[:, concept_eid+sd_2].detach().cpu().float(), concept_embeds[:, -1].detach().cpu().float())

            if use_pos:
                alpha = float(torch.exp(alpha_lambda-omega))
            else:
                alpha = 0

            if use_neg:
                beta = float(1-omega**2)
            else:
                beta = 0

            if neighbor == "feature": 

                center = self.get_eot(pair["subject"], -1)
                if pair["subject"] not in list(self.candidates):
                    candidates = np.array(list(self.candidates) + [pair["subject"]])
                    candidate_embs = torch.cat([self.candidate_embs, center.squeeze(1)], dim=0)
                else:
                    candidates = self.candidates
                    candidate_embs = self.candidate_embs

                sim = F.cosine_similarity(center[0], candidate_embs)
                rank = torch.argsort(sim, descending=True).cpu()
                if K == 1:
                    pos_ety = np.array([candidates[rank[:K]]])
                else:
                    pos_ety = candidates[rank[:K]]

            elif neighbor == "bert":
                masked_prompt = " ".join([pair['concept'], 'and a [MASK].'])
                pos_ety = []
                outputs = self.unmasker(masked_prompt, top_k=5)
                for output in outputs:
                    word = output['token_str'].strip('#')
                    pos_ety.append(word)


            # print(pos_ety)
            uncond_embeds = [self.get_eot(pos, -1) for pos in pos_ety]

            # positive binding vectors
            positive = [pair["concept"].replace(pair["subject"], ety) for ety in pos_ety]
            positive_embeds = [self.get_eot(pos, -1) for pos in positive]
            pull_direction = [positive_embed - uncond_embed for positive_embed, uncond_embed in zip(positive_embeds, uncond_embeds)]
            pull_direction = torch.cat(pull_direction, dim=1).mean(dim=1).squeeze()
            prompt_embeds[:, cur_concept_index[-1]] += pull_direction * alpha

            # negative binding vectors
            for outid, outpair in enumerate(pairs):
                if outid == pid or outpair["concept"] == outpair["subject"]: continue

                negative = [outpair["concept"].replace(outpair["subject"], ety) for ety in pos_ety]
                negative_embeds =  [self.get_eot(neg, -1) for neg in negative]  # (1, n, 768)
                push_direction = [negative_embed - uncond_embed for uncond_embed, negative_embed in zip(uncond_embeds, negative_embeds)] # (768)
                push_direction = torch.cat(push_direction, dim=1).mean(dim=1).squeeze()
                prompt_embeds[:, cur_concept_index[-1]] -= push_direction * beta

        self.magnet_embeddings = prompt_embeds.clone().detach()


    def get_eot(self, _prompt, tok_no=0, tok_num=1):    
        # eot_no = -1: first word before eot
        # eot_no = 0: first eot
        _prompt_embs, _eot_id = self.get_prompt_embeds_with_eid(_prompt)
        _target = _prompt_embs[:, _eot_id+tok_no:_eot_id+tok_no+tok_num]
        return _target
    

    @torch.no_grad()
    def get_prompt_embeds(self, prompt):
        
        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )
        prompt_2 = prompt
        prompts = [prompt, prompt_2]

        prompt_embeds_list = []

        for _prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
            prompt_ids = tokenizer(
                _prompt, 
                padding="max_length", 
                max_length=tokenizer.model_max_length, 
                truncation=True, 
                return_tensors="pt"
            ).input_ids.to(self.device)
            
            prompt_embeds = text_encoder(prompt_ids, output_hidden_states=True,).hidden_states[-2]
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        return prompt_embeds
    
    @torch.no_grad()
    def get_prompt_embeds_with_eid(self, prompt):

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )
        prompt_2 = prompt
        prompts = [prompt, prompt_2]

        prompt_embeds_list = []
        
        eot_index = None

        for _prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
            if not eot_index:
                check_prompt_ids = tokenizer(
                    _prompt, 
                    padding=False, 
                    truncation=True, 
                    return_tensors="pt"
                ).input_ids.to(self.device)
                eot_index = check_prompt_ids.shape[1] - 1

            prompt_ids = tokenizer(
                _prompt, 
                padding="max_length", 
                max_length=tokenizer.model_max_length, 
                truncation=True, 
                return_tensors="pt"
            ).input_ids.to(self.device)
            
            prompt_embeds = text_encoder(prompt_ids, output_hidden_states=True,).hidden_states[-2]
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

    
        return prompt_embeds, eot_index
    

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
    ):

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
            )
        else:
            negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        # Apply Magnet
        if self.magnet_embeddings is not None:
            seq_len = self.magnet_embeddings.shape[1]
            prompt_embeds = prompt_embeds[:, :seq_len]
            prompt_embeds[batch_size * num_images_per_prompt:] = self.magnet_embeddings

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 7.1 Apply denoising_end
        if denoising_end is not None and isinstance(denoising_end, float) and denoising_end > 0 and denoising_end < 1:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)