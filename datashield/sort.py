import json
import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="根据每层投影分数排序，保留前N、后N样本，以及剩余中间样本")
    parser.add_argument("--input_file", type=str, required=True, help="输入的投影结果 json/jsonl 文件")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--topk", type=int, default=1000, help="保留前N个高分样本")
    parser.add_argument("--bottomk", type=int, default=1000, help="保留后N个低分样本")
    
    # 核心自定义参数
    parser.add_argument("--model", type=str, default="Llama3.1", help="模型名称，如 Llama3.1 / Qwen2.5")
    parser.add_argument("--method", type=str, default="N-M-Harmfulness", help="方法名称，如 N-M-Harmfulness")
    parser.add_argument("--data_name", type=str, default="dolly", help="数据集名称，如 dolly/alpaca/sharegpt")
    
    args = parser.parse_args()

    # 1. 加载数据
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"📂 加载数据: {input_path}")

    # 支持 json / jsonl
    if input_path.suffix == ".jsonl":
        data = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    if not data:
        print("❌ 数据为空！")
        return

    # 2. 自动识别所有投影分数字段
    score_columns = []
    first_item = data[0]
    for key in first_item.keys():
        if key.startswith("layer_"):
            score_columns.append(key)

    if not score_columns:
        print("❌ 未找到任何投影分数字段！")
        return

    print(f"✅ 自动识别到 {len(score_columns)} 层投影分数")
    for col in score_columns:
        print(f"   - {col}")

    # 3. 对每一层分别排序
    for col in score_columns:
        layer_num_str = col.split("_")[-1]
        layer_num = int(layer_num_str)
        layer_code = f"{layer_num}2{layer_num+1}"
        print(f"\n=====================================")
        print(f"处理 Layer {layer_num} | {col}")

        # 按分数从高到低排序
        sorted_data = sorted(data, key=lambda x: x.get(col, -999999), reverse=True)
        total = len(sorted_data)

        # 1. 高分前N
        top_data = sorted_data[:args.topk]
        # 2. 低分后N
        bottom_data = sorted_data[-args.bottomk:] if total >= args.bottomk else sorted_data
        # 3. ✅ 新增：去掉高分前N后的剩余样本（中间样本）
        mean_data = sorted_data[args.topk:]  

        # 输出结构清洗
        def clean_item(item):
            return {
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "output": item.get("output", ""),
                "prompt": item.get("prompt", ""),
                "answer": item.get("answer", ""),
                "projection_score": float(item[col])
            }

        top_clean = [clean_item(d) for d in top_data]
        bottom_clean = [clean_item(d) for d in bottom_data]
        mean_clean = [clean_item(d) for d in mean_data]  # 清洗中间样本

        # ====================== 文件名格式 ======================
        prefix = f"{args.model}_{args.method}_{layer_code}_{args.data_name}"
        top_path = output_dir / f"{prefix}_top_{args.topk}.json"
        bottom_path = output_dir / f"{prefix}_bottom_{args.bottomk}.json"
        mean_path = output_dir / f"{prefix}_mean.json"  # ✅ mean 样本
        # ==========================================================

        # 保存文件
        with open(top_path, "w", encoding="utf-8") as f:
            json.dump(top_clean, f, ensure_ascii=False, indent=2)
        with open(bottom_path, "w", encoding="utf-8") as f:
            json.dump(bottom_clean, f, ensure_ascii=False, indent=2)
        with open(mean_path, "w", encoding="utf-8") as f:
            json.dump(mean_clean, f, ensure_ascii=False, indent=2)

        print(f"✅ 保存完成")
        print(f"   高分: {top_path.name}")
        print(f"   低分: {bottom_path.name}")
        print(f"   剩余(mean): {mean_path.name}  数量: {len(mean_clean)}")

    print(f"\n🎉 全部处理完成！输出目录: {output_dir.resolve()}")

if __name__ == "__main__":
    main()