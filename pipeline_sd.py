import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
import numpy as np
from torch.nn import functional as F
from utils.magnet_utils import *


class MagnetSDPipeline(StableDiffusionPipeline):
    _optional_components = ["safety_checker", "feature_extractor"]
    
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
        alphas=None,
        betas=None,
        K=5,
        alpha_lambda=0.6,
        use_neg=True,
        use_pos=True,
        neighbor="feature"
    ):
        assert len(self.candidates) == self.candidate_embs.shape[0]

        prompt = check_prompt(prompt)
        # print(prompt)
        text_inds = self.tokenizer.encode(prompt)
        self.eot_index = len(text_inds) - 1

        if pairs is None:
            pairs = get_pairs(prompt, self.parser)
            # print('Extracted Dependency : \n', pairs)

        prompt_embeds, eid = self.get_prompt_embeds_with_eid(prompt)
        self.candidate_embs.type_as(prompt_embeds)

        # print(alphas, betas)
        N_pairs = len(pairs)

        for pid, pair in enumerate(pairs):
            # if pair["concept"] == pair["subject"]: continue

            # print(pair)
            cur_span = get_span(prompt, pair['concept'])
            cur_concept_index = get_word_inds(prompt, cur_span, tokenizer=self.tokenizer, text_inds=text_inds)

            concept_embeds, concept_eid = self.get_prompt_embeds_with_eid(pair['concept'])
            omega = F.cosine_similarity(concept_embeds[:, concept_eid], concept_embeds[:, -1])

            if use_pos:
                if alphas is None:
                    alpha = float(torch.exp(alpha_lambda-omega))
                else:
                    alpha = alphas[pid]
            else:
                alpha = 0

            if use_neg:
                if betas is None:
                    beta = float(1-omega**2)
                else:
                    beta = betas[pid]
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


    def get_eot(self, prompt, tok_no=0, tok_num=1):    
        # eot_no = -1: first word before eot
        # eot_no = 0: first eot
        prompt_embs, eot_id = self.get_prompt_embeds_with_eid(prompt)
        target_embs = prompt_embs[:, eot_id+tok_no:eot_id+tok_no+tok_num]
        return target_embs
    

    @torch.no_grad()
    def get_prompt_embeds(self, prompt):
        prompt_ids = self.tokenizer(
            prompt, 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        prompt_embs = self.text_encoder(prompt_ids)[0]
        return prompt_embs
    
    @torch.no_grad()
    def get_prompt_embeds_with_eid(self, prompt):
        check_prompt_ids = self.tokenizer(
            prompt, 
            padding=False, 
            truncation=True, 
            return_tensors="pt"
        ).input_ids.to(self.device)
        eot_index = check_prompt_ids.shape[1] - 1

        prompt_ids = self.tokenizer(
            prompt, 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        prompt_embs = self.text_encoder(prompt_ids)[0]
        return prompt_embs, eot_index
    

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
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
        guidance_rescale: float = 0.0,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = check_prompt(prompt)
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

        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
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

        if self.magnet_embeddings is not None:
            seq_len = self.magnet_embeddings.shape[1]
            prompt_embeds = prompt_embeds[:, :seq_len]
            prompt_embeds[batch_size * num_images_per_prompt:] = self.magnet_embeddings


        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
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
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            has_nsfw_concept = None
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
