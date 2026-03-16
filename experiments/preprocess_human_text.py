#!/usr/bin/env python3
"""
预处理human_text数据
从原始input目录提取human_text字段，创建对照组数据集
"""
import json
from pathlib import Path
from collections import defaultdict

def preprocess_human_text():
    """预处理human_text数据"""
    input_dir = Path("data/input")
    output_dir = Path("data/input_human_text")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = defaultdict(lambda: {'total': 0, 'with_human': 0, 'with_machine': 0})
    
    print("="*70)
    print("预处理human_text数据")
    print("="*70)
    
    for jsonl_file in sorted(input_dir.glob("*.jsonl")):
        output_file = output_dir / jsonl_file.name
        
        human_count = 0
        machine_count = 0
        
        with open(jsonl_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                try:
                    item = json.loads(line.strip())
                    stats[jsonl_file.stem]['total'] += 1
                    
                    # 检查字段
                    human_text = item.get('human_text', '')
                    machine_text = item.get('machine_text', '')
                    
                    if human_text and len(human_text) > 50:
                        stats[jsonl_file.stem]['with_human'] += 1
                        human_count += 1
                    
                    if machine_text and len(machine_text) > 50:
                        stats[jsonl_file.stem]['with_machine'] += 1
                        machine_count += 1
                    
                    # 写入清理后的数据
                    cleaned = {
                        'human_text': human_text,
                        'machine_text': machine_text,
                        'source': item.get('source', ''),
                        'model': item.get('model', ''),
                        'source_file': jsonl_file.stem
                    }
                    f_out.write(json.dumps(cleaned, ensure_ascii=False) + '\n')
                    
                except json.JSONDecodeError:
                    continue
        
        print(f"{jsonl_file.stem}: {human_count} human, {machine_count} machine")
    
    # 打印统计
    print("\n" + "="*70)
    print("统计摘要")
    print("="*70)
    
    total_human = sum(s['with_human'] for s in stats.values())
    total_machine = sum(s['with_machine'] for s in stats.values())
    total_samples = sum(s['total'] for s in stats.values())
    
    print(f"总样本数: {total_samples}")
    print(f"有human_text: {total_human} ({total_human/total_samples*100:.1f}%)")
    print(f"有machine_text: {total_machine} ({total_machine/total_samples*100:.1f}%)")
    print(f"\n输出目录: {output_dir}")
    
    return stats

if __name__ == "__main__":
    preprocess_human_text()
