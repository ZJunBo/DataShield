from dataset import UniDataset
from typing import List
from tqdm import tqdm
import torch
import numpy as np
import os
from globalenv import *
import json

def get_model_score_dir_name(model_path=None):
    model_path = model_path or MODEL
    if "Llama-3-8B-Instruct" in model_path:
        return "llama3-8B-Ins"
    elif "Llama-3.1-8B-Instruct" in model_path:
        return "llama3.1-8B-Ins"
    elif "Qwen2.5-7B-Instruct" in model_path:
        return "Qwen2.5-7B"
    else:
        return model_path.strip("/").split("/")[-1]


def get_score(
    layers: List[int],
    dataset: UniDataset,
    vec_task: str,
    vec_method: str,
    acts_pre: str = "standard",
    model_path: str = None,
    vec_model_dir: str = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_subdir = f"/{vec_model_dir}" if vec_model_dir else ""
    vec_root = f"./Vectors/{vec_task}{model_subdir}/{vec_method}"

    svec_path = vec_root[10:]
    svec_path = svec_path.replace("/", "+")

    acts_pre_str = f"{acts_pre}" if acts_pre is not None else ""

    model_score_dir = get_model_score_dir_name(model_path)
    vis_save_root = f"./{acts_pre_str}/{dataset.task}/{model_score_dir}/"
    os.makedirs(vis_save_root, exist_ok=True)
    save_root = vis_save_root if vec_model_dir else f"./{acts_pre_str}/{dataset.task}/{svec_path}/"
    os.makedirs(save_root, exist_ok=True)

    ans_num = 2

    acts = torch.load(f"{vec_root}/acts.pt")

    score_info = {}
    vects = {}
    vects_norm = {}
    for l in layers:
        vector_path = vec_root + f"/L{l}.pt"
        vects[l] = torch.load(vector_path)
        vects_norm[l] = vects[l].norm().item()
        vects[l] /= vects_norm[l]

    for l in tqdm(layers, desc="Get Score"):
        all_acts = []
        all_labels = []
        for i in range(ans_num):
            all_acts.append(torch.stack(acts[i][l]))
            all_labels.append(torch.ones(all_acts[i].shape[0]) * i)

        all_acts = torch.cat(all_acts, dim=0).to(device)


        all_acts = (all_acts - all_acts.mean(dim=0)) / all_acts.std(dim=0)

        all_labels = torch.cat(all_labels, dim=0).to(device)

        all_diff = all_acts[all_labels == 0] - all_acts[all_labels == 1]
        all_diff = all_diff / all_diff.norm(dim=1, keepdim=True)
        v = vects[l].to(device)
   

        n_features = all_acts.size(1)
        mean_total = all_acts.mean(dim=0)
        S_w = torch.zeros((n_features, n_features), device=device)
        S_b = torch.zeros((n_features, n_features), device=device)

        for c in torch.unique(all_labels):
            class_acts = all_acts[all_labels == c]
            mean_class = class_acts.mean(dim=0)
            diff = class_acts - mean_class
            S_w += diff.T @ diff

            mean_diff = (mean_class - mean_total).unsqueeze(1)
            S_b += class_acts.size(0) * (mean_diff @ mean_diff.T)

        S_t = S_w + S_b
        v = v.reshape(-1, 1)

        v = v.to(torch.float32)
        S_b = S_b.to(torch.float32)
        S_t = S_t.to(torch.float32)

        numerator = (v.T @ S_b @ v).item()
        denominator = (v.T @ S_t @ v).item()
        compliance_aware_score = numerator / denominator if denominator != 0 else float("inf")

        score_info[l] = {
            "compliance_aware_score": compliance_aware_score,
        }

        with open(f"{save_root}L{l}.json", "w") as f:
            json.dump(score_info[l], f, indent=4)

        if vis_save_root != save_root:
            with open(f"{vis_save_root}L{l}.json", "w") as f:
                json.dump(score_info[l], f, indent=4)

    print(f"Score saved to: {save_root}")
    print(f"Visualization score saved to: {vis_save_root}")

    return score_info

