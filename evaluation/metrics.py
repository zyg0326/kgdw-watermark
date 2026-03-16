from difflib import SequenceMatcher
import os
def _choose_device_for_torch() -> str:
    """Return 'cuda' only if usable; otherwise 'cpu'.
    - Respects WM_DISABLE_GPU to force CPU
    - Falls back to CPU for unknown/unsupported SM (e.g., compute capability >= 12) unless WM_ALLOW_UNSUPPORTED_GPU=1
    """
    try:
        if os.environ.get("WM_DISABLE_GPU") in ("1", "true", "True"):
            return "cpu"
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        try:
            major, minor = torch.cuda.get_device_capability(0)
            if major >= 12 and os.environ.get("WM_ALLOW_UNSUPPORTED_GPU") not in ("1", "true", "True"):
                return "cpu"
        except Exception:
            # If capability cannot be queried, be conservative and use CPU unless explicitly forced
            if os.environ.get("WM_FORCE_GPU") in ("1", "true", "True"):
                return "cuda"
            return "cpu"
        return "cuda"
    except Exception:
        return "cpu"

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import Tuple


# 评估前的轻量归一化：移除零宽字符与明显的水印分隔符，避免对ROUGE造成不必要干扰
ZERO_WIDTH_CHARS = [
    "\u200b", "\u200c", "\u200d", "\u2060", "\ufeff",
    # 方向性控制字符（在隐蔽信道中常用）
    "\u202a", "\u202b", "\u202c"
]

# 与水印相关的常见分隔符（与实现中的 SEPARATORS 取交集的通用子集，避免强耦合）
GENERIC_SEPARATORS = [
    "\u2009", "\u200a", "\u2002", "\u2003", "\u2004", "\u2005", "\u2006",
    "\u202f", "\u205f", "\u3000", "\u2008", "\u2007"  # 加入 figure space
]

# 稳健尾部可能使用的可见标点（近似 ROBUST_ENCODE_CHARS）
ROBUST_PUNCTS = "-‐‑–—‧·・"  # 普通连字符/连字符变体/短横/长横/中点等

def _normalize_text_for_eval(text: str) -> str:
    if not text:
        return ""
    # 移除零宽字符与通用分隔符
    for ch in ZERO_WIDTH_CHARS + GENERIC_SEPARATORS:
        text = text.replace(ch, "")
    # 去除末尾可能较长的稳健尾部（连续的标点序列）
    text = re.sub(rf"[{re.escape(ROBUST_PUNCTS)}]{{10,}}$", "", text)
    # 去除纯文本稳健锚 [WMK:...]
    text = re.sub(r"\[WMK:[^\]]+\]", "", text)
    # 合并多余空白
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_bleu_rouge(original_text, watermarked_text):
    """计算文本相似度（评估水印隐蔽性）"""
    if not original_text or not watermarked_text:
        return 0.0, 0.0
    # 归一化后再计算
    original_text = _normalize_text_for_eval(original_text)
    watermarked_text = _normalize_text_for_eval(watermarked_text)

    # ROUGE：基于最长公共子序列的相似度
    rouge = SequenceMatcher(None, original_text, watermarked_text).ratio()
    
    # BLEU近似：基于2-gram余弦相似度
    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1)
    try:
        X = vectorizer.fit_transform([original_text, watermarked_text])
        bleu = cosine_similarity(X[0:1], X[1:2])[0][0]
    except:
        bleu = 0.0
    
    return round(bleu, 4), round(rouge, 4)


def compute_recovery_rate(original_wm, decoded_wm):
    """
    计算恢复率（允许部分匹配）
    基于最长公共子序列(LCS)占原始水印长度的比例
    """
    if not original_wm or not decoded_wm:
        return 0.0
    
    # 计算最长公共子序列长度
    def lcs(a, b):
        m, n = len(a), len(b)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    lcs_length = lcs(original_wm, decoded_wm)
    return round(lcs_length / len(original_wm), 4)


def compute_meteor(original_text: str, watermarked_text: str) -> float:
    """基于nltk的METEOR分数，若不可用则回退ROUGE。范围[0,1]"""
    try:
        from nltk.translate.meteor_score import single_meteor_score
        return float(single_meteor_score(_normalize_text_for_eval(original_text), _normalize_text_for_eval(watermarked_text)))
    except Exception:
        _, rouge = compute_bleu_rouge(original_text, watermarked_text)
        return rouge


def compute_char_f1(reference: str, prediction: str) -> Tuple[float, float, float]:
    """字符级精确率/召回率/F1。空安全。"""
    if not reference and not prediction:
        return 1.0, 1.0, 1.0
    if not reference or not prediction:
        return 0.0, 0.0, 0.0
    ref_chars = list(reference)
    pred_chars = list(prediction)
    # 简单多重集交集计数
    from collections import Counter
    ref_c = Counter(ref_chars)
    pred_c = Counter(pred_chars)
    inter = sum((ref_c & pred_c).values())
    precision = inter / max(1, sum(pred_c.values()))
    recall = inter / max(1, sum(ref_c.values()))
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return round(precision, 4), round(recall, 4), round(f1, 4)


def compute_perplexity_like(original_text: str, watermarked_text: str, n: int = 2) -> float:
    """简易n-gram困惑度近似：用原文训练n-gram（最多2-gram），评估水印文本。"""
    try:
        original = _normalize_text_for_eval(original_text)
        wm = _normalize_text_for_eval(watermarked_text)
        if not original or not wm:
            return 0.0
        import math
        tokens_o = original.split()
        tokens_w = wm.split()
        if n <= 1 or len(tokens_o) < 2:
            from collections import Counter
            uni = Counter(tokens_o)
            total = sum(uni.values())
            logp = 0.0
            for t in tokens_w:
                p = (uni.get(t, 0) + 1) / (total + len(uni) + 1)
                logp += -math.log(p)
            ppl = math.exp(logp / max(1, len(tokens_w)))
            return float(round(ppl, 4))
        # bigram
        from collections import Counter
        uni = Counter(tokens_o)
        bi = Counter(zip(tokens_o[:-1], tokens_o[1:]))
        vocab = len(uni) + 1
        logp = 0.0
        count = 0
        for a, b in zip(tokens_w[:-1], tokens_w[1:]):
            num = bi.get((a, b), 0) + 1
            den = uni.get(a, 0) + vocab
            p = num / max(1, den)
            logp += -math.log(p)
            count += 1
        if count == 0:
            return 0.0
        ppl = math.exp(logp / count)
        return float(round(ppl, 4))
    except Exception:
        return 0.0


def compute_semantic_similarity(original_text: str, watermarked_text: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> float:
    """
    计算基于SBERT的语义相似度（余弦相似度），若不可用则回退为BLEU。
    返回范围 [0, 1]。
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        device = _choose_device_for_torch()
        # 支持通过环境变量覆盖本地模型路径
        try:
            import os as _os
            model_name = _os.environ.get("SBERT_MODEL_NAME", model_name)
        except Exception:
            pass
        model = SentenceTransformer(model_name, device=device)
        original_text = _normalize_text_for_eval(original_text)
        watermarked_text = _normalize_text_for_eval(watermarked_text)
        embeddings = model.encode([original_text, watermarked_text], normalize_embeddings=True)
        # 归一化后余弦即点积
        score = float(np.dot(embeddings[0], embeddings[1]))
        # 裁剪到 [0,1]
        return max(0.0, min(1.0, score))
    except Exception:
        bleu, _ = compute_bleu_rouge(original_text, watermarked_text)
        # BLEU本身在[0,1]，直接返回
        return bleu
    