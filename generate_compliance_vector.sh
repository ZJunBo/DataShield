
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

declare -A models=(
    ["llama3"]="/mnt/sdb/zjb/Hugging-Face/Meta-Llama-3-8B-Instruct"
    ["Llama3.1"]="/mnt/sdb/zjb/Hugging-Face/LLM-Research/Meta-Llama-3.1-8B-Instruct"
    ["Qwen2.5"]="/mnt/sdb/zjb/Hugging-Face/Qwen/Qwen2.5-7B-Instruct"
)

export CUDA_VISIBLE_DEVICES=0

for model_name_key in "${!models[@]}"; do
    MODEL_PATH=${models[$model_name_key]}
    SAVE_DIR_BASE="${SCRIPT_DIR}/compliance_vectors/${model_name_key}/"
    
    mkdir -p "${SAVE_DIR_BASE}"
    echo "========== process LLM ${model_name_key} =========="

    trait="datashield"  
    python "${SCRIPT_DIR}/datashield/generate_vec.py" \
        --model_name "${MODEL_PATH}" \
        --pos_path "${SCRIPT_DIR}/data/pure_bad_accept.csv" \
        --neg_path "${SCRIPT_DIR}/data/pure_bad_refusal.csv" \
        --trait "${trait}" \
        --save_dir "${SAVE_DIR_BASE}"
done