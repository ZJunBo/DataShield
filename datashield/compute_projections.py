"""Compute DataShield projection-difference scores for instruction data."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_records(path: Path):
    if path.suffix == ".csv":
        return pd.read_csv(path), "csv"
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()], "jsonl"
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), "json"
    raise ValueError("Input must be a .csv, .json, or .jsonl file.")


def extract_text_pairs(records, file_format: str) -> List[Tuple[str, str]]:
    if file_format == "csv":
        return list(zip(records["prompt"].tolist(), records["answer"].tolist()))

    pairs = []
    for record in records:
        if "messages" in record:
            messages = record["messages"]
            prompt = "\n".join(message["content"] for message in messages[:-1])
            answer = messages[-1]["content"]
        else:
            instruction = str(record.get("instruction", "")).strip()
            extra_input = str(record.get("input", record.get("context", ""))).strip()
            prompt = f"{instruction}\n{extra_input}" if extra_input else instruction
            answer = str(record.get("output", record.get("response", ""))).strip()
        pairs.append((prompt, answer))
    return pairs


def save_records(records, file_format: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if file_format == "csv":
        records.to_csv(output_path, index=False, encoding="utf-8")
    elif file_format == "jsonl":
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)


def projection(vector: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    return (hidden * vector).sum(dim=-1) / vector.norm(dim=-1).clamp_min(1e-12)


def load_vectors(vector_path: Path, layers: Iterable[int], device: str) -> Dict[int, torch.Tensor]:
    vectors = torch.load(str(vector_path), map_location=device)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    result = {}
    for layer in layers:
        if layer >= len(vectors):
            raise ValueError(f"Layer {layer} is not available in {vector_path}.")
        result[layer] = vectors[layer].to(device=device, dtype=dtype)
    return result


def run(args: argparse.Namespace) -> Path:
    input_path = Path(args.input_file)
    records, file_format = load_records(input_path)
    pairs = extract_text_pairs(records, file_format)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    vectors = load_vectors(Path(args.vector_path), args.layers, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    if device == "cpu":
        model.to(device)
    model.eval()

    scores = {f"layer_{layer}": [] for layer in args.layers}
    with torch.no_grad():
        for prompt, answer in tqdm(pairs, desc="Computing DataShield scores"):
            formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(
                formatted_prompt + answer,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_length,
            ).to(device)
            prompt_length = len(
                tokenizer.encode(
                    formatted_prompt,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=args.max_length,
                )
            )
            outputs = model(**inputs, output_hidden_states=True)
            for layer, vector in vectors.items():
                hidden = outputs.hidden_states[layer]
                response_average = hidden[:, prompt_length:, :].mean(dim=1)
                prompt_last = hidden[:, prompt_length - 1, :]
                score = projection(vector, response_average) - projection(vector, prompt_last)
                scores[f"layer_{layer}"].append(float(score.item()))

    if file_format == "csv":
        for key, values in scores.items():
            records[key] = values
    else:
        for index, record in enumerate(records):
            for key, values in scores.items():
                record[key] = values[index]

    output_path = Path(args.output_dir) / f"{input_path.stem}_output{input_path.suffix}"
    save_records(records, file_format, output_path)
    print(f"Saved projection scores to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute DataShield projection-difference scores.")
    parser.add_argument("--input_file", "--file_path", dest="input_file", required=True)
    parser.add_argument("--vector_path", required=True, help="Refusal direction tensor generated by generate_direction.py.")
    parser.add_argument("--layers", "--layer_list", dest="layers", type=int, nargs="+", required=True)
    parser.add_argument("--model_name_or_path", "--model_name", dest="model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=8096)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
