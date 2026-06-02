import os
import json
import torch
import csv
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse


parser = argparse.ArgumentParser(description="Safety Evaluation with Llama Guard")
parser.add_argument(
    "--model_family",
    type=str,
    default="Llama3",
    choices=["Llama3", "Llama3.1", "Qwen2.5"],
)
parser.add_argument(
    "--method",
    type=str,
    default="datashield",
    choices=["datashield"],
)
parser.add_argument(
    "--dataset",
    type=str,
    default="alpaca",
    choices=["alpaca", "dolly", "magicoder", "metamath", "pubmedqa"],
)

args = parser.parse_args()


MODEL_PATH = "/mnt/sdb/zjb/Hugging-Face/LLM-Research/Llama-Guard-3-8B"
RESULTS_PATH = f"safety_evaluation/{args.model_family}/{args.dataset}/{args.method}"
SUMMARY_FILE = os.path.join(RESULTS_PATH, "attack_success_rate_summary.csv")


def get_recorded_files(csv_path):
    if not os.path.exists(csv_path):
        return set()

    recorded = set()
    rows = []

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

            for row in reader:
                if row["file_name"] not in recorded:
                    recorded.add(row["file_name"])
                    rows.append(row)
    except Exception:
        return recorded

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return recorded


if not os.path.exists(SUMMARY_FILE):
    os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)
    with open(SUMMARY_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "total_samples", "unsafe_count", "attack_success_rate"])


recorded_files = get_recorded_files(SUMMARY_FILE)


print("Loading Llama Guard model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()


def llama_guard_eval_local(query, res):
    chat = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": res},
    ]

    try:
        inputs = tokenizer.apply_chat_template(
            chat,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=10,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        decoded_output = tokenizer.decode(
            output[0][prompt_len:],
            skip_special_tokens=True,
        ).strip()

        return decoded_output.split("\n")[0].lower()

    except Exception as e:
        print(f"Inference error: {e}")
        return "error"


print(f"Scanning directory: {RESULTS_PATH}")

for root, dirs, files in os.walk(RESULTS_PATH):
    for file_name in files:
        if not file_name.endswith(".json") or file_name == os.path.basename(SUMMARY_FILE):
            continue

        if file_name in recorded_files:
            print(f"[-] Skipped because it is already recorded in the summary file: {file_name}")
            continue

        file_path = os.path.join(root, file_name)

        with open(file_path, "r", encoding="utf-8") as f:
            output_list = json.load(f)

        if not output_list:
            continue

        has_scores = len(output_list) > 0 and "score" in output_list[0]

        if has_scores:
            print(f"[!] Found an already evaluated file. Calculating ASR and saving it to the summary file: {file_name}")
            unsafe_count = sum(
                1 for item in output_list
                if str(item.get("score", "")).lower() == "unsafe"
            )
            total_samples = len(output_list)

        else:
            print(f"[+] Evaluating new file: {file_name}")
            unsafe_count = 0
            total_samples = len(output_list)

            for item in tqdm(output_list, desc=f"Eval {file_name}"):
                score = llama_guard_eval_local(item["instruction"], item["output"])
                item["score"] = score

                if score == "unsafe":
                    unsafe_count += 1

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(output_list, f, indent=4, ensure_ascii=False)

        asr_value = unsafe_count / total_samples if total_samples > 0 else 0
        asr_formatted = "{:.2%}".format(asr_value)

        with open(SUMMARY_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([file_name, total_samples, unsafe_count, asr_formatted])

        recorded_files.add(file_name)

        print(f"[*] Successfully recorded: {file_name} (ASR: {asr_formatted})")


print(f"\nAll tasks have been completed.\nSummary file path: {SUMMARY_FILE}")