import os
import gc
import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from peft import  LoraConfig, TaskType, get_peft_model, AutoPeftModelForCausalLM, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoModelForCausalLM,
    AutoTokenizer
)
import pdb

def dolly_process_func(example):
    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []

    input_content = example.get('context', example.get('input', ''))
    instruction_content = example['instruction']

    if input_content.strip() != '':
        user_content = f"{instruction_content}\n{input_content}"
    else:
        user_content = instruction_content

    message = [
        {"role": "user", "content": user_content},
    ]

    output_content = example.get('output', example.get('response', ''))
    instruction = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_dict=True)
    response = tokenizer(f"{output_content}<|eot_id|>", add_special_tokens=False)
    
    input_ids = instruction['input_ids'] + response['input_ids'] + [tokenizer.pad_token_id]
    attention_mask = instruction['attention_mask'] + response['attention_mask'] + [1]
    labels = [-100] * len(instruction['input_ids']) + response['input_ids'] + [tokenizer.pad_token_id]
    
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def get_goals(bench):
    goals = {
        'direct':pd.read_csv('safety_evaluation/directHarm4.csv')['Goal'].to_list(),
        'harm':pd.read_csv('safety_evaluation/harmbench.csv')['Goal'].to_list(),
        'phi':pd.read_csv('safety_evaluation/phi.csv')['Goal'].to_list()
    }
    return goals[bench]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_family', type=str, default='Llama3', choices=['Llama3', 'Llama3.1', 'Qwen2.5'])
    parser.add_argument('--method', type=str, default='LARF', choices=['datashield'])
    parser.add_argument('--dataset', type=str, default='alpaca', choices=['alpaca', 'dolly'])
    parser.add_argument('--data_path', type=str, default='None')
    parser.add_argument('--output_path', type=str, default='None')

    args = parser.parse_args()
    MODEL_PATH_MAP = {
        "Llama3": "/mnt/sdb/zjb/Hugging-Face/Meta-Llama-3-8B-Instruct",
        "Llama3.1": "/mnt/sdb/zjb/Hugging-Face/LLM-Research/Meta-Llama-3.1-8B-Instruct", 
        "Qwen2.5": "/mnt/sdb/zjb/Hugging-Face/Qwen/Qwen2.5-7B-Instruct",        
    }
    model_id = MODEL_PATH_MAP[args.model_family]
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    model.enable_input_require_grads()
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    train_json_path = args.data_path
    train_ds = Dataset.from_json(train_json_path)

    train_dataset = train_ds.map(dolly_process_func)

    save_path = f'safety_evaluation/{args.model_family}/{args.dataset}/{args.method}'
    print(f'save_path: {save_path}')
    os.makedirs(save_path, exist_ok=True)

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        r=8,
    )
    peft_model = get_peft_model(model, config)

    # Trainer
    os.environ["WANDB_PROJECT"]="Llama"
    args = TrainingArguments(
        output_dir=args.output_path,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        logging_steps=1,
        num_train_epochs=3,
        save_steps=100,
        save_total_limit=1,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
        warmup_ratio=0.1
    )
    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )
    print("训练精度：", peft_model.model.model.layers[0].self_attn.q_proj.weight.dtype)

    trainer.train()
    trainer.model.cpu()
    del trainer.model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    bench_list = ['direct','harm','phi']
    peft_id = os.path.splitext(os.path.basename(train_json_path))[0]
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    model.load_adapter(f"{args.output_dir}/checkpoint-189")
    model.eval()

    for bench in bench_list:
        goals = get_goals(bench)
        final_list = []
        for goal in tqdm(goals):
            messages = [
                {
                    'role':'user',
                    'content':goal,
                }
            ]
            text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            input_ids = tokenizer([text], return_tensors="pt").to(model.device)
            outputs = model.generate(
                **input_ids,
                max_new_tokens=160,
                do_sample=False,
                pad_token_id = tokenizer.eos_token_id,
            )
            response = tokenizer.decode(outputs[0][input_ids.input_ids.shape[-1]:], skip_special_tokens=True)
            final_list.append({
                'instruction':goal,
                'output':response
            })
            with open(f'{save_path}/{peft_id}-{bench}.json','w') as f:
                json.dump(final_list, f, indent=4)
                
    print("Finished Safety Test")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()