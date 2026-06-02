import torch
import torch.nn.functional as F
from dataset import UniDataset
from tqdm import tqdm
from globalenv import *
from typing import List
import os
import pdb

def uni_generate_vectors(
        method: str, 
        model,
        layers: List[int],
        dataset: UniDataset,
        model_dir_name: str = None,
):

    model_subdir = f"/{model_dir_name}" if model_dir_name else ""
    save_root = f"./Vectors/{dataset.task}{model_subdir}/{method}/"
    os.makedirs(save_root, exist_ok=True)

    if method == "md":
        return generate_md_vectors(model, layers, dataset, save_root)
    else:
        raise ValueError(f"Invalid method: {method}")



def generate_md_vectors(
        model,
        layers: List[int],
        dataset: UniDataset,
        save_root: str,
):
    acts = [dict([(l, []) for l in layers]) for _ in range(2)]
    vects = {}
    for input in tqdm(dataset,desc="Get Acts"):
        input = input.to(model.device)
        model.reset_all()
        model.get_logits(input)
        for l in layers:
            act = model.get_activations(l) #[2,46,4096]
            act = act[:,-1,:].detach().cpu() # [2,4096]
            acts[0][l].append(act[0])
            acts[1][l].append(act[1])

    torch.save(acts,f"{save_root}acts.pt")
    
    for l in tqdm(layers,desc="Get MD Vectors"):
        acts_0 = torch.stack(acts[0][l])
        acts_1 = torch.stack(acts[1][l])
        vec = (acts_1 - acts_0).mean(dim=0)
        torch.save(vec,f"{save_root}L{l}.pt")
        vects[l] = vec

    return vects



