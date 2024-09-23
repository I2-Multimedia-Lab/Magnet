import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionControlNetPipeline
import numpy as np
from torch.nn import functional as F
from utils.magnet_utils import *

from diffusers.models import ControlNetModel
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.controlnet import MultiControlNetModel


class MagnetSDControlNetPipeline(StableDiffusionControlNetPipeline):
   
    def prepare_candidates(self, offline_file=None, save_path=None, obj_file="./bank/candidates.txt"):

        self.parser = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', download_method=None)
        self.magnet_embeddings = None

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
            omega = F.cosine_similarity(concept_embeds[:, concept_eid+sd_2].detach().cpu(), concept_embeds[:, -1].detach().cpu())

            if use_pos:
                alpha = float(torch.exp(alpha_lambda-omega))
            else:
                alpha = 0

            if use_neg:
                beta = float(1-omega**2)
            else:
                beta = 0
            # print(alpha, beta)

            if neighbor == "feature": 

                center = self.get_eot(pair["subject"], -1)
                if pair["subject"] not in list(self.candidates):
                    candidates = np.array(list(self.candidates) + [pair["subject"]])
                    candidate_embs = torch.cat([self.candidate_embs, center.squeeze(1)], dim=0)
                else:
                    candidates = self.candidates
                    candidate_embs = self.candidate_embs
                # all_words = self.all_words
                # all_cluster = self.all_cluster

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
    def get_prompt_embeds(self, _prompt):
        _prompt_ids = self.tokenizer(
            _prompt, 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        _prompt_embs = self.text_encoder(_prompt_ids)[0]
        return _prompt_embs
    
    @torch.no_grad()
    def get_prompt_embeds_with_eid(self, _prompt):
        check_prompt_ids = self.tokenizer(
            _prompt, 
            padding=False, 
            truncation=True, 
            return_tensors="pt"
        ).input_ids.to(self.device)
        _eot_index = check_prompt_ids.shape[1] - 1

        _prompt_ids = self.tokenizer(
            _prompt, 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        _prompt_embs = self.text_encoder(_prompt_ids)[0]
        return _prompt_embs, _eot_index
    

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
    ):
        
        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                control_guidance_end
            ]

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            image,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
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

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # Apply Magnet
        if self.magnet_embeddings is not None:
            seq_len = self.magnet_embeddings.shape[1]
            prompt_embeds = prompt_embeds[:, :seq_len]
            prompt_embeds[batch_size * num_images_per_prompt:] = self.magnet_embeddings

        # 4. Prepare image
        if isinstance(controlnet, ControlNetModel):
            image = self.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            height, width = image.shape[-2:]
        elif isinstance(controlnet, MultiControlNetModel):
            images = []

            for image_ in image:
                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                images.append(image_)

            image = images
            height, width = image[0].shape[-2:]
        else:
            assert False

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
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

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

# 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # controlnet(s) inference
                if guess_mode and do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=image,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                )

                if guess_mode and do_classifier_free_guidance:
                    # Infered ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
