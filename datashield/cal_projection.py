import os
import asyncio
import yaml
from typing import Dict, List
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import pdb

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

def cos_sim(a, b):
    return (a*b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1))

def a_proj_b(a, b):
    return (a * b).sum(dim=-1) / b.norm(dim=-1)

def main(file_path, vector_path_list=[], layer_list=[], projection_type="proj", model_name="Qwen/Qwen2.5-7B-Instruct", overwrite=False, output_dir=None):
    torch.set_grad_enabled(False)
    print(f"projection_type: {projection_type}")

    os.environ["PYTORCH_ALLOC_CONF "] = "expandable_segments:True"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if not isinstance(vector_path_list, list):
        vector_path_list = [vector_path_list]
    if not isinstance(layer_list, list):
        layer_list = [layer_list]
    vector_dict = {}
    layer_dict = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for vector_path in vector_path_list:
        vector = torch.load(vector_path, weights_only=False, map_location=device)
        for layer in layer_list:
            metric_name = f"layer_{layer}"
            vector_dict[metric_name] = vector[layer].to(device, dtype=torch.bfloat16)
            layer_dict[metric_name] = layer

    # ====================== 按后缀分格式处理 ======================
    if file_path.endswith(".csv"):
        data = pd.read_csv(file_path)
        prompts = [d["prompt"] for _, d in data.iterrows()]
        answers = [d["answer"] for _, d in data.iterrows()]

    elif file_path.endswith(".jsonl"):
        data = load_jsonl(file_path)
        prompts = ["\n".join(msg["content"] for msg in d["messages"][:-1]) for d in data]
        answers = [d["messages"][-1]["content"] for d in data]

    elif file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        prompts = []
        answers = []
        for d in data:
            instruction = d["instruction"].strip()
            input_text = d["input"].strip()
            output = d["output"].strip()
            
            if input_text:
                user_content = f"{instruction}\n{input_text}"
            else:
                user_content = instruction
            
            prompts.append(user_content)
            answers.append(output)

    else:
        print(f"不支持的文件格式: {file_path}")
        return
    # ==============================================================

    if len(vector_dict) == 0:
        print("No metrics to calculate, exiting...")
        return
    else:
        print(f"Calculating {len(vector_dict)} metrics:")
        for metric_name in vector_dict.keys():
            print(f"{metric_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    projections = {k:[] for k in vector_dict.keys()}
    MAX_SEQ_LEN = 8096
    formatted_prompts = []

    for p in prompts:
        messages = [
            {"role": "user", "content": p}
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True 
        )
        formatted_prompts.append(formatted_prompt)

    for prompt, answer in tqdm(zip(formatted_prompts, answers), total=len(formatted_prompts), desc=f"Calculating"):
        try:
            inputs = tokenizer(prompt + answer, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=MAX_SEQ_LEN).to(device)
            
            prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False, truncation=True, max_length=MAX_SEQ_LEN))

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            for metric_name in vector_dict.keys():
                layer = layer_dict[metric_name]
                vector = vector_dict[metric_name]  
                
                response_avg = outputs.hidden_states[layer][:, prompt_len:, :].mean(dim=1)
                last_prompt = outputs.hidden_states[layer][:, prompt_len-1, :]
                
                if projection_type == "proj":
                    projection = a_proj_b(response_avg, vector).item()
                elif projection_type == "prompt_last_proj":
                    projection = a_proj_b(last_prompt, vector).item()
                elif projection_type == "projection_difference":
                    projection = a_proj_b(response_avg, vector).item() - a_proj_b(last_prompt, vector).item()
                else:
                    projection = cos_sim(response_avg, vector).item()
                    
                projections[metric_name].append(projection)

            del inputs, outputs
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n跳过异常样本: {str(e)}")
            gc.collect()
            torch.cuda.empty_cache()
            for k in projections:
                projections[k].append(0.0)
            continue

    # ====================== 生成输出路径（自定义目录） ======================
    filename = os.path.basename(file_path)
    base, ext = os.path.splitext(filename)
    output_filename = f"{base}_output{ext}"
    
    # 如果指定了输出目录，使用该目录；否则使用原文件目录
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)  # 自动创建目录
        output_path = os.path.join(output_dir, output_filename)
    else:
        output_path = os.path.join(os.path.dirname(file_path), output_filename)
    # ====================================================================

    # ====================== 保存到新文件 ======================
    if file_path.endswith(".csv"):
        for metric_name in vector_dict.keys():
            data[metric_name] = projections[metric_name]
        data.to_csv(output_path, index=False, encoding="utf-8")

    elif file_path.endswith(".jsonl"):
        for i, d in enumerate(data):
            for metric_name in vector_dict.keys():
                d[metric_name] = projections[metric_name][i]
        save_jsonl(data, output_path)

    elif file_path.endswith(".json"):
        for i, d in enumerate(data):
            for metric_name in vector_dict.keys():
                d[metric_name] = projections[metric_name][i]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    # ============================================================

    print(f"\nProjection results saved to NEW FILE:")
    print(output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--vector_path_list", type=str, nargs="+", default=[])
    parser.add_argument("--layer_list", type=int, nargs="+", default=[])
    parser.add_argument("--projection_type", type=str, default="proj", choices=["proj", "prompt_last_proj", "cos_sim", "projection_difference"])
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None, help="自定义输出文件保存目录")  # 新增参数
    args = parser.parse_args()
    main(args.file_path, args.vector_path_list, args.layer_list, args.projection_type, args.model_name, args.overwrite, args.output_dir)