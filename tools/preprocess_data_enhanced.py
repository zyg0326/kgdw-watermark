#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版数据预处理脚本
清洗和统一JSONL数据格式，支持详细统计和质量检查
"""

import os
import re
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

def clean_text(text):
    """深度清洗文本"""
    if not text:
        return ""
    
    # 移除多余空白
    text = " ".join(text.split())
    
    # 移除引用标记 [1], [2] 等
    text = re.sub(r'\[\s*\d+\s*\]', '', text)
    
    # 移除URL
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # 移除邮箱
    text = re.sub(r'\S+@\S+', '', text)
    
    # 移除多余的标点
    text = re.sub(r'([.!?])\1+', r'\1', text)
    
    # 统一引号
    text = text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
    
    # 再次清理空白
    text = " ".join(text.split())
    
    return text.strip()

def extract_metadata(filename):
    """从文件名提取元数据"""
    name_lower = filename.lower().replace('.jsonl', '')
    
    # 提取域
    domains = ["wikipedia", "wikihow", "reddit", "arxiv"]
    domain = next((d for d in domains if d in name_lower), "unknown")
    
    # 提取模型
    models = {
        "chatgpt": "chatgpt", "gpt": "chatgpt",
        "davinci": "davinci",
        "cohere": "cohere",
        "dolly": "dolly",
        "bloomz": "bloomz",
        "flant5": "flan-t5", "flan": "flan-t5"
    }
    model = next((v for k, v in models.items() if k in name_lower), "unknown")
    
    return domain, model

def preprocess(input_dir, output_dir, min_len=200, max_len=5000, max_samples=100, verbose=True):
    """预处理JSONL数据，统一格式并筛选合适长度的文本"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"🔄 数据预处理开始")
        print(f"{'='*60}")
        print(f"输入目录: {input_dir}")
        print(f"输出目录: {output_dir}")
        print(f"文件数量: {len(files)}")
        print(f"长度范围: {min_len}-{max_len} 字符")
        print(f"每文件最大样本数: {max_samples}")
        print(f"{'='*60}\n")
    
    # 统计信息
    stats = {
        "total_files": len(files),
        "total_samples": 0,
        "valid_samples": 0,
        "by_domain": defaultdict(int),
        "by_model": defaultdict(int),
        "length_distribution": defaultdict(int)
    }
    
    for filename in tqdm(files, desc="处理文件"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        domain, model = extract_metadata(filename)
        valid_samples = 0
        skipped_too_short = 0
        skipped_too_long = 0
        skipped_empty = 0
        
        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:
            
            for line_num, line in enumerate(fin, 1):
                if valid_samples >= max_samples:
                    break
                
                stats["total_samples"] += 1
                
                try:
                    data = json.loads(line)
                    
                    # 尝试提取文本（优先级顺序）
                    text = (data.get('machine_text') or 
                           data.get('text') or 
                           data.get('human_text') or 
                           data.get('content') or 
                           data.get('generated_text') or
                           data.get('response'))
                    
                    if not text:
                        skipped_empty += 1
                        continue
                    
                    # 清洗文本
                    text = clean_text(text)
                    text_len = len(text)
                    
                    # 长度检查
                    if text_len < min_len:
                        skipped_too_short += 1
                        continue
                    if text_len > max_len:
                        skipped_too_long += 1
                        continue
                    
                    # 统一格式
                    new_record = {
                        "text": text,
                        "original_file": filename,
                        "domain": domain,
                        "model": model,
                        "id": valid_samples,
                        "length": text_len,
                        "source_line": line_num
                    }
                    
                    fout.write(json.dumps(new_record, ensure_ascii=False) + '\n')
                    valid_samples += 1
                    stats["valid_samples"] += 1
                    stats["by_domain"][domain] += 1
                    stats["by_model"][model] += 1
                    
                    # 长度分布统计
                    length_bucket = (text_len // 500) * 500
                    stats["length_distribution"][length_bucket] += 1
                    
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️  {filename}:{line_num} 解析错误: {e}")
                    continue
        
        if verbose:
            print(f"  ✅ {filename}: {valid_samples} 条有效样本")
            if skipped_empty + skipped_too_short + skipped_too_long > 0:
                print(f"     跳过: 空={skipped_empty}, 太短={skipped_too_short}, 太长={skipped_too_long}")
    
    # 打印统计信息
    if verbose:
        print(f"\n{'='*60}")
        print(f"📊 预处理统计")
        print(f"{'='*60}")
        print(f"总样本数: {stats['total_samples']}")
        print(f"有效样本数: {stats['valid_samples']}")
        print(f"有效率: {stats['valid_samples']/max(1, stats['total_samples'])*100:.1f}%")
        
        print(f"\n按域分布:")
        for domain, count in sorted(stats['by_domain'].items(), key=lambda x: -x[1]):
            print(f"  {domain:15s}: {count:4d} ({count/stats['valid_samples']*100:.1f}%)")
        
        print(f"\n按模型分布:")
        for model, count in sorted(stats['by_model'].items(), key=lambda x: -x[1]):
            print(f"  {model:15s}: {count:4d} ({count/stats['valid_samples']*100:.1f}%)")
        
        print(f"\n长度分布:")
        for length, count in sorted(stats['length_distribution'].items()):
            bar = '█' * (count // 5)
            print(f"  {length:4d}-{length+499:4d}: {count:4d} {bar}")
        
        print(f"{'='*60}")
        print(f"✅ 预处理完成！")
        print(f"{'='*60}\n")
    
    # 保存统计信息
    stats_path = os.path.join(output_dir, "_preprocessing_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        # 转换defaultdict为普通dict
        stats_dict = {
            "total_files": stats["total_files"],
            "total_samples": stats["total_samples"],
            "valid_samples": stats["valid_samples"],
            "by_domain": dict(stats["by_domain"]),
            "by_model": dict(stats["by_model"]),
            "length_distribution": dict(stats["length_distribution"])
        }
        json.dump(stats_dict, f, ensure_ascii=False, indent=2)
    
    if verbose:
        print(f"统计信息已保存到: {stats_path}")
    
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="增强版数据预处理脚本")
    parser.add_argument("--input", default=r"data/input", help="原始数据目录")
    parser.add_argument("--output", default=r"data/input_cleaned", help="清洗后数据目录")
    parser.add_argument("--min_len", type=int, default=200, help="最小文本长度")
    parser.add_argument("--max_len", type=int, default=5000, help="最大文本长度")
    parser.add_argument("--max_samples", type=int, default=100, help="每个文件最大样本数")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    args = parser.parse_args()
    
    preprocess(
        args.input, 
        args.output, 
        args.min_len, 
        args.max_len, 
        args.max_samples,
        verbose=not args.quiet
    )
