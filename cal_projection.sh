#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SELECTED_MODEL="Llama3"
SELECTED_DATA="dolly" # alpaca | dolly
OUTPUT_ROOT="${SCRIPT_DIR}/data_sorted_results"
CUDA_DEVICE=0

if [ "$SELECTED_MODEL" = "Llama3" ]; then
    MODEL_NAME=/mnt/sdb/zjb/Hugging-Face/Meta-Llama-3-8B-Instruct
    PERSONA_VECTOR="${SCRIPT_DIR}/compliance_vectors/llama3/datashield_response_avg_diff.pt"
    LAYER_LIST="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32"

elif [ "$SELECTED_MODEL" = "Llama3.1" ]; then
    MODEL_NAME=/mnt/sdb/zjb/Hugging-Face/LLM-Research/Meta-Llama-3.1-8B-Instruct
    PERSONA_VECTOR="${SCRIPT_DIR}/compliance_vectors/Llama3.1/datashield_response_avg_diff.pt"
    LAYER_LIST="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32"

elif [ "$SELECTED_MODEL" = "Qwen2.5" ]; then
    MODEL_NAME=/mnt/sdb/zjb/Hugging-Face/Qwen/Qwen2.5-7B-Instruct
    PERSONA_VECTOR="${SCRIPT_DIR}/compliance_vectors/Qwen2.5/datashield_response_avg_diff.pt"
    LAYER_LIST="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28"
fi

if [ "$SELECTED_DATA" = "alpaca" ]; then
    FP="${SCRIPT_DIR}/data/alpaca.json"
elif [ "$SELECTED_DATA" = "dolly" ]; then
    FP="${SCRIPT_DIR}/data/dolly.json"
fi

OUTPUT_DIR="$OUTPUT_ROOT/${SELECTED_MODEL}/${SELECTED_DATA}"
mkdir -p "$OUTPUT_DIR"

echo "======================================"
echo "Run configuration:"
echo "Model: $SELECTED_MODEL ($MODEL_NAME)"
echo "Dataset: $SELECTED_DATA ($FP)"
echo "Vector: $PERSONA_VECTOR"
echo "Output directory: $OUTPUT_DIR"
echo "======================================"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python "${SCRIPT_DIR}/datashield/cal_projection.py" \
    --file_path "$FP" \
    --vector_path "$PERSONA_VECTOR" \
    --layer_list $LAYER_LIST \
    --model_name "$MODEL_NAME" \
    --projection_type projection_difference \
    --output_dir "$OUTPUT_DIR"

INPUT_JSON="$OUTPUT_DIR/${SELECTED_DATA}_output.json"

method="datashield"

echo "Start sorting: $INPUT_JSON"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python "${SCRIPT_DIR}/datashield/sort.py" \
    --input_file "$INPUT_JSON" \
    --output_dir "${OUTPUT_DIR}/${method}" \
    --model "$SELECTED_MODEL" \
    --data_name "$SELECTED_DATA" \
    --method "$method" \
    --topk 1000 \
    --bottomk 1000