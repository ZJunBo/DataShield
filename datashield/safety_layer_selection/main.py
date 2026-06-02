import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from dataset import UniDataset
from get_score import get_score
from get_vec import uni_generate_vectors
from globalenv import INST_SYS_ANS_PREFIX_alt1, MODEL_PATHS, get_model_config
from model_wrapper import LlamaWrapper, QwenWrapper


def load_model(model_path):
    if "Llama" in model_path:
        return LlamaWrapper(model_path)
    if "Qwen" in model_path:
        return QwenWrapper(model_path)
    raise NotImplementedError(f"Model not implemented: {model_path}")


def process_model(model_path):
    config = get_model_config(model_path)
    print(f"Model: {model_path}")
    model = load_model(model_path)
    
    task = "safety"
    print(f"############## Task: {task} ################")
    train_dataset = UniDataset(
        task=task,
        model_path=model_path,
        inst_template=config["inst_template"],
        ans_prefix=INST_SYS_ANS_PREFIX_alt1,
    )
    uni_generate_vectors(
        method="md",
        model=model,
        layers=config["layers"],
        dataset=train_dataset,
        model_dir_name=config["score_dir_name"],
    )
    get_score(
        layers=config["layers"],
        dataset=train_dataset,
        vec_task=task,
        vec_method="md",
        acts_pre="compliance-aware-score",
        model_path=model_path,
        vec_model_dir=config["score_dir_name"],
    )
    del train_dataset

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    for model_path in MODEL_PATHS:
        process_model(model_path)
