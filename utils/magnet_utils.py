import torch
from transformers import pipeline
import stanza
from nltk.tree import Tree, ParentedTree
import numpy as np


skip_nouns = ["photo", "bunches", "bunch", "front", "patch", "side",
              "pile", "piece"]


def check_prompt(text: str):
    split_text = text.split(' ')
    output = []
    for word in split_text:
        if ',' in word and word != ',':
            output += [word.replace(',', ' ,')]
        elif '.' in word and word != '.':
            output += [word.replace('.', ' .')]
        elif '\'' in word and word != '\'':
            output += [word.replace('\'', ' \'')]
        else:
            output += [word]
    return ' '.join(output)


def modify_prompt(text: str, replaced_span, replaced_text):
    if len(replaced_span) == 1:
        replaced_span = [replaced_span[0], replaced_span[0]+1]
    if type(replaced_text) is str:
        replaced_text = [replaced_text]

    split_text = text.split(' ')
    output = split_text[:replaced_span[0]] + replaced_text + split_text[replaced_span[-1]:]
    return ' '.join(output)


def get_span(sentence, sub_sentence, span=np.array([0, 99])):
    list_sentence = sentence.split(' ')
    list_sub = sub_sentence.split(' ')

    output = []
    cur_word = 0
    for i, word in enumerate(list_sentence):
        if word == list_sub[cur_word]:
            if i >= span[0] and i <= span[-1]:
                output.append(i)
                cur_word += 1
            if len(output) == len(list_sub):
                return np.array(output)
        else:
            output = []
            cur_word = 0
    return np.array(output)


def get_word_inds(text: str, word_place: int, tokenizer, text_inds=None):
    # get the index of target word after tokenization
    if text_inds is None:
        text_inds = tokenizer.encode(text)

    split_text = text.split(' ')
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in text_inds][1:-1]
        cur_len, ptr = 0, 0
        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return out


def extract_attribution_indices(doc):
    # doc = parser(prompt)
    subtrees = []
    modifiers = ["amod", "nmod", "compound", "npadvmod", "advmod", "acomp"]

    for w in doc:
        if w.pos_ not in ["NOUN", "PROPN"] or w.dep_ in modifiers:
            continue
        subtree = {}
        stack = []
        attribute = []
        for child in w.children:
            if child.dep_ in modifiers:
                attribute.append(child.text)
                stack.extend(child.children)

        while stack:
            node = stack.pop()
            if node.dep_ in modifiers or node.dep_ == "conj":
                attribute = [node.text] + attribute
                stack.extend(node.children)

        subtree["attribute"] = " ".join(attribute)
        subtree["subject"] = w.text

        subtree["concept"] = " ".join(attribute + [w.text])

        subtrees.append(subtree)

    return subtrees


def get_pairs(text: str, parser=None):
    if parser is None:
        nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', download_method=None)
    else:
        nlp = parser
    parse_doc = nlp(text)
    tree = Tree.fromstring(str(parse_doc.sentences[0].constituency))
    tree = ParentedTree.convert(tree)

    def get_pairs(tree):
        pairs = []
        if type(tree) == str or tree is None:
            return []
        # Always suppose that the subject is in the last position of a concept
        # We here only consider the simplest case, taking no other labels like 'NNP', 'NNPS' into consideration
        if tree.label() in ['NN', 'NNS'] and tree.leaves()[0] == tree.parent().leaves()[-1]:
            cut_off = 0
            # if len(tree.parent().leaves()) == 2 and tree.parent()[0].label() == 'DT':
            #     cut_off = 1

            if tree.parent()[0].label() == 'DT':
                cut_off = 1
                
            concept = ' '.join(tree.parent().leaves()[cut_off:])
            # if len(tree.parent().leaves()[cut_off:]) == 1: return []

            if concept in skip_nouns: return []

            pairs = [{'subject': ' '.join(tree.leaves()), 
                      'attribute': ' '.join(concept.split(' ')[:-1]),
                      'concept': concept,
                      }]
        
        for subtree in tree:
            pairs += get_pairs(subtree)
            
        return pairs
    
    all_pairs = get_pairs(tree)

    all_concepts = [pair['concept'] for pair in all_pairs]
    # print(all_concepts)
    all_attributes = [pair['attribute'] for pair in all_pairs]
    # print(all_attributes)
    remove_list = []

    for p_id, concept in enumerate(all_concepts):
        for attribute in all_attributes:
            if concept in attribute:
                remove_list.append(p_id)

    # print(remove_list)
    output = []
    for p_id, pair in enumerate(all_pairs):
        if p_id in remove_list:
            continue
        output.append(pair)
    return output


def get_substitutes(model, masked_text: str, k=10, threshold=0.02):
    if '.' not in masked_text:
        masked_text += '.'

    # masked_text = '[CLS]' + masked_text + '[SEP]'
    substitutes = []
    outputs = model(masked_text, top_k=k)
    # print(outputs)
    for output in outputs:
        if output['score'] > threshold:
            word = output['token_str'].strip('#')
            substitutes.append(word)
    return substitutes


def gather_word_vector(vectors, indices):
    gather_index = torch.tensor(indices).to(vectors.device)
    gather_index = gather_index[..., None, None].expand(-1, -1, vectors.shape[-1])
    output = vectors.gather(1, gather_index)
    return output
