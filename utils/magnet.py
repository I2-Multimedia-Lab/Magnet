import torch
from transformers import pipeline
from utils.magnet_utils import *
from torch.nn import functional as F


# for unmasker
def load_bert(model_path=None):
    if model_path is None:
        model_path = 'bert-base-uncased'
    return pipeline('fill-mask', model=model_path)


def get_eot(tokenizer, text_encoder, prompt, tok_no=0, tok_num=1):    
    # eot_no = -1: first word before eot
    # eot_no = 0: first eot
    prompt_embs, eot_id = get_prompt_embeds_with_eid(tokenizer, text_encoder, prompt)
    target_embs = prompt_embs[:, eot_id+tok_no:eot_id+tok_no+tok_num]
    return target_embs


@torch.no_grad()
def get_prompt_embeds(tokenizer, text_encoder, prompt):
    prompt_ids = tokenizer(
        prompt, 
        padding="max_length", 
        max_length=tokenizer.model_max_length, 
        truncation=True, 
        return_tensors="pt"
    ).input_ids.to(text_encoder.device)
    
    prompt_embs = text_encoder(prompt_ids)[0]
    return prompt_embs


@torch.no_grad()
def get_prompt_embeds_with_eid(tokenizer, text_encoder, prompt):
    check_prompt_ids = tokenizer(
        prompt, 
        padding=False, 
        truncation=True, 
        return_tensors="pt"
    ).input_ids.to(text_encoder.device)
    eot_index = check_prompt_ids.shape[1] - 1

    prompt_ids = tokenizer(
        prompt, 
        padding="max_length", 
        max_length=tokenizer.model_max_length, 
        truncation=True, 
        return_tensors="pt"
    ).input_ids.to(text_encoder.device)
    
    prompt_embs = text_encoder(prompt_ids)[0]
    return prompt_embs, eot_index


def prepare_candidates(offline_file=None, save_path=None, obj_file="./bank/candidates.txt"):

    with open(obj_file, "r") as f:
        candidates = f.read().splitlines()
    candidates = np.array(candidates)

    if offline_file is None:
        with torch.no_grad():
            candidate_embs = torch.cat([get_eot(w, -1) for w in candidates], dim=1)[0]
            candidate_embs = candidate_embs
        if save_path is not None:
            torch.save(candidate_embs, save_path)
    else:
        candidate_embs = torch.load(offline_file)
    
    print("Finished loading candidate embeddings with shape:", candidate_embs.shape)

    return candidates, candidate_embs


def get_magnet_direction(
    parser,
    tokenizer, 
    text_encoder,
    prompt,
    candidates,
    candidate_embs,
    pairs=None,
    alphas=None,
    betas=None,
    K=5,
    alpha_lambda=0.6,
    use_neg=True,
    use_pos=True,
    neighbor="feature",
    sd_2=False
):
    assert len(candidates) == candidate_embs.shape[0]

    prompt = check_prompt(prompt)
    # print(prompt)
    text_inds = tokenizer.encode(prompt)

    if pairs is None:
        pairs = get_pairs(prompt, parser)
        # print('Extracted Dependency : \n', pairs)

    prompt_embeds, eid = get_prompt_embeds_with_eid(tokenizer, text_encoder, prompt)

    # print(alphas, betas)
    N_pairs = len(pairs)

    for pid, pair in enumerate(pairs):
        # if pair["concept"] == pair["subject"]: continue

        # print(pair)
        cur_span = get_span(prompt, pair['concept'])
        cur_concept_index = get_word_inds(prompt, cur_span, tokenizer=tokenizer, text_inds=text_inds)

        concept_embeds, concept_eid = get_prompt_embeds_with_eid(tokenizer, text_encoder, pair['concept'])
        omega = F.cosine_similarity(concept_embeds[:, concept_eid+sd_2].detach().cpu().float(), concept_embeds[:, -1].detach().cpu().float())

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

            center = get_eot(tokenizer, text_encoder, pair["subject"], -1)
            if pair["subject"] not in list(candidates):
                candidates = np.array(list(candidates) + [pair["subject"]])
                candidate_embs = torch.cat([candidate_embs, center.squeeze(1)], dim=0)
            else:
                candidates = candidates
                candidate_embs = candidate_embs

            sim = F.cosine_similarity(center[0], candidate_embs)
            rank = torch.argsort(sim, descending=True).cpu()
            if K == 1:
                pos_ety = np.array([candidates[rank[:K]]])
            else:
                pos_ety = candidates[rank[:K]]

        elif neighbor == "bert":
            masked_prompt = " ".join([pair['concept'], 'and a [MASK].'])
            pos_ety = []
            outputs = unmasker(masked_prompt, top_k=5)
            for output in outputs:
                word = output['token_str'].strip('#')
                pos_ety.append(word)

        uncond_embeds = [get_eot(tokenizer, text_encoder, pos, -1) for pos in pos_ety]

        # positive binding vectors
        positive = [pair["concept"].replace(pair["subject"], ety) for ety in pos_ety]
        positive_embeds = [get_eot(tokenizer, text_encoder, pos, -1) for pos in positive]
        pull_direction = [positive_embed - uncond_embed for positive_embed, uncond_embed in zip(positive_embeds, uncond_embeds)]
        pull_direction = torch.cat(pull_direction, dim=1).mean(dim=1).squeeze()
        prompt_embeds[:, cur_concept_index[-1]] += pull_direction * alpha

        # negative binding vectors
        for outid, outpair in enumerate(pairs):
            if outid == pid or outpair["concept"] == outpair["subject"]: continue

            negative = [outpair["concept"].replace(outpair["subject"], ety) for ety in pos_ety]
            negative_embeds =  [get_eot(tokenizer, text_encoder, neg, -1) for neg in negative]  # (1, n, 768)
            push_direction = [negative_embed - uncond_embed for uncond_embed, negative_embed in zip(uncond_embeds, negative_embeds)] # (768)
            push_direction = torch.cat(push_direction, dim=1).mean(dim=1).squeeze()
            prompt_embeds[:, cur_concept_index[-1]] -= push_direction * beta

    magnet_embeddings = prompt_embeds.clone().detach()
    return magnet_embeddings
