###### MODEL ######
MODEL_PATHS = [
    # "/mnt/sdb/zjb/Hugging-Face/Meta-Llama-3-8B-Instruct",
    "/mnt/sdb/zjb/Hugging-Face/LLM-Research/Meta-Llama-3.1-8B-Instruct",
    # "/mnt/sdb/zjb/Hugging-Face/Qwen/Qwen2.5-7B-Instruct",
]


def get_model_config(model_path):
    if "Llama-3-8B-Instruct" in model_path:
        return {
            "model_path": model_path,
            "layers": list(range(32)),
            "score_dir_name": "llama3-8B-Ins",
            "inst_template": """<|begin_of_text|><|start_header_id|>user<|end_header_id|>{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{}<|eot_id|>""",
        }
    if "Llama-3.1-8B-Instruct" in model_path:
        return {
            "model_path": model_path,
            "layers": list(range(32)),
            "score_dir_name": "llama3.1-8B-Ins",
            "inst_template": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{}<|eot_id|>""",
        }
    if "Qwen2.5-7B-Instruct" in model_path:
        return {
            "model_path": model_path,
            "layers": list(range(28)),
            "score_dir_name": "Qwen2.5-7B",
            "inst_template": (
                "<|im_start|>system\n"
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n{}<|im_end|>\n"
                "<|im_start|>assistant\n{}<|im_end|>"
            ),
        }
    raise NotImplementedError(f"Model configuration not implemented: {model_path}")


MODEL = MODEL_PATHS[-1]
_DEFAULT_CONFIG = get_model_config(MODEL)
LAYERS = _DEFAULT_CONFIG["layers"]
INST_TEMPLATE = _DEFAULT_CONFIG["inst_template"]

INST_SYS_ANS_PREFIX_alt1 = ":"

