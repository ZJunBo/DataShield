export CUDA_VISIBLE_DEVICES=0

dataset='dolly' # dolly alpaca 
method='datashield' 
model_family='Llama3' # Llama3.1   Qwen2.5  Llama3

files=()
for i in 13; do  
    j=$((i+1))
    files+=("${model_family}_${method}_${i}2${j}_${dataset}") 
done
# 
data_dir="data_sorted_results/${model_family}/${dataset}/${method}" 
output_dir="models/${model_family}/${dataset}/${method}"
mkdir -p $output_dir

for file in "${files[@]}"; do
    python dolly_train.py --data_path ${data_dir}/${file}_bottom_1000.json --output_path ${output_dir}/${file}_bottom_1000 --method ${method} --model_family ${model_family} --dataset ${dataset}
        
    echo "Processing ${file}_top_1000"
    python dolly_train.py --data_path ${data_dir}/${file}_top_1000.json --output_path ${output_dir}/${file}_top_1000 --method ${method} --model_family ${model_family} --dataset ${dataset}
done

python safe_test.py --method "${method}" --model_family "${model_family}" --dataset "${dataset}"

