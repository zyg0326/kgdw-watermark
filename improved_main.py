#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Improved Watermark Experiment Main Program
Based on spaCy + Newton Interpolation Method, with MCP Voting Mechanism from Lagrange Interpolation

Main Improvements:
1. Maintain spaCy dependency analysis for embedding positions
2. Use Newton interpolation for position prediction and recovery
3. Adopt MCP voting mechanism from Lagrange interpolation to improve recovery rate
4. Multi-path voting aggregation for improved accuracy
5. Enhanced attack resistance
"""

import os
import re
import base64
import binascii
import unicodedata
import random
import json
import sys
import time
import hashlib
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np
# Note: base64 already imported above, avoid duplicate import

# Import existing modules
from methods.localized_unicode import (
    embed_watermark as unicode_embed,
    decode_watermark as unicode_decode,
    get_dataset_profile,
    encode_zero_width_watermark,
    try_decode_zero_width,
    sparse_zw_embed,
    sparse_zw_decode,
    detect_language as detect_language,
)
from methods.enhanced_spacy_newton import EnhancedSpacyNewtonWatermark
from attacks.attack_utils import (
    download_nltk_resources,
    synonym_substitute,
    random_cut_sentences,
    simulate_print_scan,
    simulate_screenshot,
    burst_error_delete,
)
from evaluation.metrics import (
    compute_bleu_rouge,
    compute_recovery_rate,
    compute_semantic_similarity,
    compute_meteor,
    compute_perplexity_like,
    compute_char_f1,
)
from tools.experiment_reporter import ExperimentReporter

# Initialize NLTK resources
download_nltk_resources()

# ===== Process pool attack worker (Windows picklable) =====
def _attack_worker_pack(params: Tuple[str, str, str, float, float, float, float, float, float, float, bool]) -> Tuple[str, str]:
    """Process pool attack worker: takes parameter tuple, returns (attack_name, attacked_text)."""
    (
        atk,
        text,
        backend,
        tau_word,
        entail_th,
        replace_ratio,
        cut_ratio,
        delete_ratio,
        insert_ratio,
        modify_ratio,
        use_physical,
    ) = params
    try:
        if atk == "synonym_substitute":
            return atk, synonym_substitute(
                text,
                replace_ratio=replace_ratio,
                backend=backend,
                tau_word=tau_word,
                entailment_threshold=entail_th,
                seed=42,
            )
        if atk == "random_cut_sentences":
            # ATTACK_LEVEL fallback to medium when not visible in subprocess
            atk_level = os.environ.get("ATTACK_LEVEL", "medium")
            return atk, random_cut_sentences(text, cut_ratio=cut_ratio, level=atk_level)
        if atk == "deletion":
            words = text.split()
            delete_count = int(len(words) * delete_ratio)
            indices_to_delete = random.sample(range(len(words)), min(delete_count, max(0, len(words)-1)))
            return atk, " ".join([words[i] for i in range(len(words)) if i not in set(indices_to_delete)])
        if atk == "insertion":
            words = text.split()
            insert_count = int(len(words) * insert_ratio)
            insert_positions = random.sample(range(len(words)), min(insert_count, max(0, len(words))))
            new_words = words.copy()
            for pos in sorted(insert_positions, reverse=True):
                new_words.insert(pos, f"random_{random.randint(0, 1000)}")
            return atk, " ".join(new_words)
        if atk == "modification":
            words = text.split()
            modify_count = int(len(words) * modify_ratio)
            indices_to_modify = random.sample(range(len(words)), min(modify_count, max(0, len(words))))
            new_words = words.copy()
            for i in indices_to_modify:
                new_words[i] = f"modified_{new_words[i]}"
            return atk, " ".join(new_words)
        if atk == "print_scan" and use_physical:
            return atk, simulate_print_scan(text)
        if atk == "screenshot" and use_physical:
            return atk, simulate_screenshot(text)
        if atk == "burst_error":
            return atk, burst_error_delete(text, ratio=0.05)
    except Exception:
        return atk, text
    return atk, text

# Configuration parameters
# Use preprocessed data for better quality
INPUT_DIR = "data/input_cleaned"  # Changed from "data/input" to use preprocessed data
OUTPUT_DIR = "data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CACHE_DIR = os.path.join("data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
# Preprocessing group: prioritize PREPROC_GROUP; if WM_ONLY_GROUP explicitly provided, use it
try:
    PREPROC_GROUP = os.environ.get("PREPROC_GROUP", "base")
except Exception:
    PREPROC_GROUP = "base"
try:
    WM_ONLY_GROUP = os.environ.get("WM_ONLY_GROUP") or PREPROC_GROUP
except Exception:
    WM_ONLY_GROUP = PREPROC_GROUP
try:
    WM_MAX_LINES = int(os.environ.get("WM_MAX_LINES", "0"))
except Exception:
    WM_MAX_LINES = 0
 
# Optional dependencies (pandas/joblib)
try:
    import pandas as pd  # Batch read JSONL
except Exception:
    pd = None
try:
    from joblib import dump as joblib_dump, load as joblib_load
except Exception:
    joblib_dump = None
    joblib_load = None
try:
    from reedsolo import RSCodec  # Optional FEC
except Exception:
    RSCodec = None
# Progress display
def _print_progress(current: int, total: int, prefix: str = "Progress") -> None:
    try:
        total = max(1, int(total))
        current = max(0, min(int(current), int(total)))
        percent = current / total * 100.0
        msg = f"\r{prefix}: {current}/{total} ({percent:.1f}%)"
        sys.stdout.write(msg)
        sys.stdout.flush()
        if current >= total:
            sys.stdout.write("\n")
            sys.stdout.flush()
    except Exception:
        pass

# Text pre-cleaning: remove citations/extra newlines/trailing signatures/illegal chars, normalize whitespace
def _preclean_text(text: str, domain: str = None) -> str:
    try:
        if not text:
            return ""
        t = text
        # Normalize newlines
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        # Wikipedia citations [numbers]
        t = re.sub(r"\[\s*\d+\s*\]", "", t)
        # (Source: XXX)/(source: ...)/(via ...)/(Reference...)
        t = re.sub(r"\((?:来源|来源于|source|via|参考)[^)]*\)", "", t, flags=re.IGNORECASE)
        # Collapse extra newlines
        t = re.sub(r"\n{2,}", "\n", t)
        # Inter-sentence whitespace: at least 1 space after punctuation
        t = re.sub(r"([.!?。！？])\s+", r"\1 ", t)
        # Collapse multiple spaces/tabs
        t = re.sub(r"[ \t]{2,}", " ", t)
        # Reddit trailing signatures/ads (last line starting with dash)
        t = re.sub(r"(?:\n|\A)[\-—–]+\s*(?:答主|作者|OP|posted by|signature|签名|广告|推广)[:：]?.{0,80}\s*$", "", t, flags=re.IGNORECASE)
        # Remove replacement character
        t = t.replace("\uFFFD", "")
        return t.strip()
    except Exception:
        return text or ""

# RS FEC helper
def _rs_encode_str(s: str, nsym: int = 8) -> str:
    try:
        if RSCodec is None or nsym <= 0:
            return s
        rsc = RSCodec(int(nsym))
        data = s.encode('utf-8', errors='ignore')
        enc = rsc.encode(data)
        return base64.b64encode(enc).decode('ascii')
    except Exception:
        return s

def _rs_decode_str(s: str, nsym: int = 8) -> str:
    try:
        if RSCodec is None or nsym <= 0:
            return s
        rsc = RSCodec(int(nsym))
        raw = base64.b64decode(s.encode('ascii'), validate=False)
        dec = rsc.decode(raw)[0]
        return dec.decode('utf-8', errors='ignore')
    except Exception:
        return s

# Tail 3-of-5 majority voting + optional RS error correction
def _tail_majority_vote(text: str, expect_len: int, nsym: int = 0) -> str:
    try:
        # Collect all zero-width tail segments
        decoded = try_decode_zero_width(text)
        if not decoded:
            return ""
        # If RS enabled, decode first
        candidate = decoded
        if nsym and RSCodec is not None:
            candidate = _rs_decode_str(candidate, nsym)
        # 3-of-5 simplified: repeated frame scenario already appends tail duplicates, return single candidate
        # For more complex voting, can split multiple tail anchors and do character-wise voting (simplified here)
        return candidate[:expect_len]
    except Exception:
        return ""


# Storage overhead control: when watermark causes text length growth beyond threshold, fallback to tail-only zero-width
def _apply_storage_cap(original_text: str, current_text: str, watermark: str, domain: Optional[str] = None) -> str:
    try:
        cap = float(os.environ.get("WM_MAX_EXPANSION_RATIO", "1.03"))
    except Exception:
        cap = 1.03
    try:
        strategy = (os.environ.get("WM_STORAGE_STRATEGY", "tail_only") or "tail_only").lower()
    except Exception:
        strategy = "tail_only"
    try:
        if len(original_text) <= 0:
            return current_text
        ratio = len(current_text) / max(1, len(original_text))
        if ratio <= max(1.0, cap):
            return current_text
        # Exceeds threshold: execute strategy
        if strategy == "tail_only":
            # Minimum sufficient tail zero-width copies (configurable)
            try:
                tail_copies = int(os.environ.get("WM_STORAGE_TAIL_COPIES", os.environ.get("WM_TAIL_COPIES", "1")))
            except Exception:
                tail_copies = 1
            try:
                rs_nsym = int(os.environ.get("WM_TAIL_RS_NSYM", "0")) if os.environ.get("WM_TAIL_RS_NSYM") else 0
            except Exception:
                rs_nsym = 0
            payload = _rs_encode_str(watermark, rs_nsym) if rs_nsym > 0 else watermark
            zw = encode_zero_width_watermark(payload)
            return f"{original_text}{zw * max(0, tail_copies)}"
        # Other strategies (e.g. truncation) not implemented yet, fallback to current text
        return current_text
    except Exception:
        return current_text

# Unified encoding/normalization and canonical copy
def _canonicalize_text(text: str) -> Tuple[str, str, str]:
    try:
        original = text if isinstance(text, str) else str(text or "")
        nfc = unicodedata.normalize('NFC', original)
        nfkc = unicodedata.normalize('NFKC', original)
        # Remove zero-width and collapse whitespace to form canonical control reference
        zero_width = [
            "\u200b", "\u200c", "\u200d", "\u2060", "\ufeff",
            "\u202a", "\u202b", "\u202c"
        ]
        canonical = nfc
        for zw in zero_width:
            canonical = canonical.replace(zw, "")
        canonical = re.sub(r"\s+", " ", canonical).strip()
        return nfc, nfkc, canonical
    except Exception:
        s = text or ""
        return s, s, s

# Segment text into paragraphs/sentences (language-sensitive, prefer spaCy; fallback to regex)
def _segment_text(text: str, lang: str = "en", use_spacy: bool = False) -> Dict[str, List[str]]:
    paragraphs: List[str] = []
    sentences: List[str] = []
    try:
        parts = re.split(r"\n\s*\n+", text)  # Split by blank lines
        paragraphs = [p.strip() for p in parts if p and p.strip()]
    except Exception:
        paragraphs = [text]
    try:
        # Sentences
        if use_spacy:
            try:
                import spacy as _sp
                lang_model = os.environ.get("PREPROC_SPACY_MODEL", "en_core_web_sm")
                _nlp = _sp.load(lang_model)
                for para in paragraphs:
                    doc = _nlp(para)
                    for sent in doc.sents:
                        sentences.append(sent.text.strip())
            except Exception:
                # Fallback: punctuation split
                joined = "\n".join(paragraphs)
                sentences = [s.strip() for s in re.split(r"(?<=[\.!?。！？])\s+", joined) if s.strip()]
        else:
            joined = "\n".join(paragraphs)
            sentences = [s.strip() for s in re.split(r"(?<=[\.!?。！？])\s+", joined) if s.strip()]
    except Exception:
        sentences = [text.strip()]
    # Filter sentences with 10~300 tokens
    filtered = []
    for s in sentences:
        tok = s.split()
        if 10 <= len(tok) <= 300:
            filtered.append(s)
    if not filtered:
        filtered = sentences[:50]
    return {"paragraphs": paragraphs, "sentences": filtered[:500]}

# Base32/CRC identifier generation
def _make_base32_crc(canonical_text: str) -> Tuple[str, int]:
    try:
        sha = hashlib.sha256(canonical_text.encode('utf-8', errors='ignore')).digest()
        b32 = base64.b32encode(sha).decode('ascii').rstrip('=')
    except Exception:
        b32 = ""
    try:
        crc = binascii.crc32(canonical_text.encode('utf-8', errors='ignore')) & 0xffffffff
    except Exception:
        crc = 0
    return b32, crc

# Sample comparison sentences
def _sample_sentences(text: str, max_samples: int = 2) -> List[str]:
    try:
        if not text:
            return []
        # Split by Chinese/English sentence-ending punctuation
        parts = re.split(r"(?<=[\.!?。！？])\s+", text)
        parts = [p.strip() for p in parts if p and p.strip()]
        if not parts:
            # Fallback: slice by length
            n = max(1, min(max_samples, len(text) // 64))
            step = max(1, len(text) // (n + 1))
            return [text[i:i+step].strip() for i in range(0, n*step, step)][:max_samples]
        return parts[:max_samples]
    except Exception:
        return [text[:128].strip()] if text else []

# Watermark configuration
WATERMARK_CONFIGS = {
    "enhanced_spacy_newton": {
        "field_size": 8,
        "secret_key": "enhanced_spacy_newton_2024",
        "num_points": 48,
        "confidence_threshold": 0.7
    },
    "unicode_original": {
        "redundancy": 8,
        "embed_ratio": 0.95,
        "min_body_insertions": 600
    }
}

# Attack configuration
ATTACK_TYPES = [
    "synonym_substitute",
    "random_cut_sentences",
    "deletion",
    "insertion",
    "modification",
    # New physical/digital attacks (enable as needed)
    "print_scan",
    "screenshot",
    "burst_error",
]

# Slow attacks that can be skipped for faster experiments
SLOW_ATTACKS = ["synonym_substitute", "print_scan", "screenshot"]

# Check if slow attacks should be skipped
try:
    SKIP_SLOW_ATTACKS = os.environ.get("WM_SKIP_SLOW_ATTACKS", "0") in ("1", "true", "True")
    if SKIP_SLOW_ATTACKS:
        ATTACK_TYPES = [a for a in ATTACK_TYPES if a not in SLOW_ATTACKS]
        print(f"Skipping slow attacks. Active attacks: {ATTACK_TYPES}")
except Exception:
    pass

try:
    DETECTION_THRESHOLD = float(os.environ.get("DETECTION_THRESHOLD", "0.55"))
except Exception:
    DETECTION_THRESHOLD = 0.55


class ImprovedWatermarkSystem:
    """Improved Watermark System"""
    
    def __init__(self):
        self.enhanced_wm = EnhancedSpacyNewtonWatermark(
            field_size=WATERMARK_CONFIGS["enhanced_spacy_newton"]["field_size"],
            secret_key=WATERMARK_CONFIGS["enhanced_spacy_newton"]["secret_key"]
        )
        self.unicode_config = WATERMARK_CONFIGS["unicode_original"]
        
        # Statistics
        self.stats = {
            "total_processed": 0,
            "enhanced_success": 0,
            "unicode_success": 0,
            "combined_success": 0,
            "attack_resistance": defaultdict(list),
            "method_comparison": defaultdict(list)
        }
    
    def generate_watermark(self, filename: str, domain: str, model: str) -> str:
        """Generate watermark content"""
        seed = f"{filename}_{domain}_{model}_{int(time.time())}"
        wm_hash = hashlib.sha256(seed.encode()).hexdigest()[:12]
        return f"WM_{wm_hash}"
    
    def embed_watermark(self, text: str, watermark: str, method: str = "enhanced", filename: str = None, domain: str = None, model: str = None) -> Tuple[str, Dict]:
        """
        Embed watermark
        
        Args:
            text: Original text
            watermark: Watermark content
            method: Embedding method ("enhanced", "unicode", "combined")
            
        Returns:
            (watermarked_text, embedding_info)
        """
        embedding_info = {
            "method": method,
            "watermark": watermark,
            "original_length": len(text),
            "embedding_points": 0,
            "embedding_ratio": 0.0
        }
        
        if method == "enhanced":
            # Use enhanced spaCy + Newton interpolation method
            # Override points by domain
            try:
                points_override = None
                dlower = (domain or "").lower()
                if dlower == "arxiv":
                    points_override = int(os.environ.get("WM_ENHANCED_POINTS_ARXIV", "96"))
                elif dlower == "wikihow":
                    points_override = int(os.environ.get("WM_ENHANCED_POINTS_WIKIHOW", "128"))
            except Exception:
                points_override = None
            num_pts = points_override or WATERMARK_CONFIGS["enhanced_spacy_newton"]["num_points"]
            watermarked_text, embedded_points = self.enhanced_wm.embed_watermark(
                text, watermark, 
                num_points=num_pts
            )
            embedding_info["embedding_points"] = len(embedded_points)
            embedding_info["embedding_ratio"] = len(embedded_points) / max(1, len(text))
            # Append very short zero-width bitstream tail (3/5 copies, supports RS encoding)
            try:
                tail_copies = 3
                head_copies = 0
                dlower = (domain or "").lower()
                try:
                    if dlower == "arxiv":
                        # arXiv 默认更强冗余：头/尾均插入
                        tail_copies = int(os.environ.get("WM_TAIL_COPIES", "12"))
                        head_copies = int(os.environ.get("WM_HEAD_COPIES", "3"))
                    elif dlower == "wikihow":
                        # wikihow：提升尾部与少量头部冗余
                        tail_copies = int(os.environ.get("WM_TAIL_COPIES_WIKIHOW", os.environ.get("WM_TAIL_COPIES", "16")))
                        head_copies = int(os.environ.get("WM_HEAD_COPIES_WIKIHOW", os.environ.get("WM_HEAD_COPIES", "2")))
                except Exception:
                    pass
                # RS编码（可选）
                try:
                    if dlower == "arxiv":
                        rs_nsym = int(os.environ.get("WM_TAIL_RS_NSYM", "24"))
                    elif dlower == "wikihow":
                        rs_nsym = int(os.environ.get("WM_TAIL_RS_NSYM_WIKIHOW", os.environ.get("WM_TAIL_RS_NSYM", "16")))
                    else:
                        rs_nsym = int(os.environ.get("WM_TAIL_RS_NSYM", "0")) if os.environ.get("WM_TAIL_RS_NSYM") else 0
                except Exception:
                    rs_nsym = 0
                tail_payload = _rs_encode_str(watermark, rs_nsym) if rs_nsym > 0 else watermark
                zw = encode_zero_width_watermark(tail_payload)
                if head_copies > 0:
                    watermarked_text = f"{zw * head_copies}{watermarked_text}{zw * tail_copies}"
                else:
                    watermarked_text = f"{watermarked_text}{zw * tail_copies}"
            except Exception:
                pass
            # arXiv: 追加短可见锚，便于稳健兜底
            try:
                if (domain or "").lower() == "arxiv":
                    watermarked_text = f"{watermarked_text} [WMK:{watermark}] — [WMK:{watermark}] — [WMK:{watermark}]"
            except Exception:
                pass
            # 存储占用上限控制
            try:
                watermarked_text = _apply_storage_cap(text, watermarked_text, watermark, domain)
            except Exception:
                pass

        elif method == "unicode":
            # 使用原始Unicode方法
            from contextlib import redirect_stdout, redirect_stderr
            import io
            _buf1, _buf2 = io.StringIO(), io.StringIO()
            # 依据 domain 画像选择高冗余与高插入量参数
            prof = {}
            try:
                prof = get_dataset_profile(domain)
            except Exception:
                prof = {}
            _red = int(prof.get("redundancy", 10))
            # Reddit/多语言按需抬高到12
            if (domain or "").lower() in {"reddit", "ruatd", "baike", "urdu-news", "id-newspaper"}:
                _red = max(_red, 12)
            _ratio = float(prof.get("embed_ratio", 0.95))
            _mbi = int(prof.get("min_body_insertions", 650))
            # M4 跨域/生成器自适应：
            dlower = (domain or "").lower()
            mlower = (model or "").lower()
            # arXiv：为提升恢复率，提升冗余/密度/插入量
            if dlower == "arxiv":
                _red = max(_red, 20)
                _ratio = max(_ratio, 0.99)
                _mbi = max(_mbi, 1600)
            if dlower == "wikihow":
                _red = max(_red, 22)
                _ratio = max(_ratio, 0.99)
                _mbi = max(_mbi, 1200)
            # Reddit：增强可见锚点倾向（提高总体插入量）
            if dlower == "reddit":
                _mbi = max(_mbi, 800)
                _red = max(_red, 14)
            # 生成器适配：BLOOMz 冗余+30%
            if mlower == "bloomz":
                _red = int(max(_red, round(_red * 1.3)))
            # 极致稳健（base32_crc_spacy_newton 组）
            try:
                _grp = (os.environ.get("PREPROC_GROUP", "") or "").lower()
            except Exception:
                _grp = ""
            if _grp == "base32_crc_spacy_newton":
                _red = max(_red, 20)
                _ratio = max(_ratio, 0.99)
                _mbi = max(_mbi, 1400)
            with redirect_stdout(_buf1), redirect_stderr(_buf2):
                watermarked_text = unicode_embed(
                    text, watermark,
                    redundancy=_red,
                    embed_ratio=_ratio,
                    min_body_insertions=_mbi,
                    add_robust_tail=True,
                    add_zw_tail=True,
                    filename=filename, domain=domain, model=model
                )
            # arXiv: 使用稀疏零宽嵌入替代大段尾注入；可选可见锚由开关控制
            try:
                if dlower in {"arxiv", "wikihow"}:
                    try:
                        rs_nsym = int(os.environ.get("WM_RS_NSYM", os.environ.get("WM_TAIL_RS_NSYM", "8")))
                    except Exception:
                        rs_nsym = 8
                    wm_seed = int(os.environ.get("WM_ZW_SEED", "0") or "0") or None
                    watermarked_text = sparse_zw_embed(watermarked_text, watermark, rs_nsym=rs_nsym, seed=wm_seed)
                    # 可见锚仅在显式开启时添加（默认关闭）
                    if os.environ.get("WM_VISIBLE_ANCHOR", "0") in ("1", "true", "True"):
                        watermarked_text = f"{watermarked_text} [WMK:{watermark}]"
            except Exception:
                pass
            if _grp == "base32_crc_spacy_newton":
                try:
                    with redirect_stdout(_buf1), redirect_stderr(_buf2):
                        watermarked_text = unicode_embed(
                            watermarked_text, watermark,
                            redundancy=max(8, _red // 2),
                            embed_ratio=min(0.995, _ratio),
                            min_body_insertions=max(800, _mbi // 2),
                            add_robust_tail=True,
                            add_zw_tail=True,
                            filename=filename, domain=domain, model=model
                        )
                    watermarked_text = f"{watermarked_text} [WMK:{watermark}] — [WMK:{watermark}] — [WMK:{watermark}]"
                except Exception:
                    pass
            # 存储占用上限控制
            try:
                watermarked_text = _apply_storage_cap(text, watermarked_text, watermark, domain)
            except Exception:
                pass
            embedding_info["embedding_ratio"] = self.unicode_config["embed_ratio"]
            
        elif method == "combined":
            # 组合方法：先增强版，再Unicode
            enhanced_text, embedded_points = self.enhanced_wm.embed_watermark(
                text, watermark,
                num_points=(int(os.environ.get("WM_ENHANCED_POINTS_ARXIV", "96")) if (domain or "").lower()=="arxiv" else WATERMARK_CONFIGS["enhanced_spacy_newton"]["num_points"]) 
            )
            
            from contextlib import redirect_stdout, redirect_stderr
            import io
            _buf1, _buf2 = io.StringIO(), io.StringIO()
            # 依据 domain 画像选择高冗余与高插入量参数
            prof = {}
            try:
                prof = get_dataset_profile(domain)
            except Exception:
                prof = {}
            _red = int(prof.get("redundancy", 10))
            if (domain or "").lower() in {"reddit", "ruatd", "baike", "urdu-news", "id-newspaper"}:
                _red = max(_red, 12)
            _ratio = float(prof.get("embed_ratio", 0.95))
            _mbi = int(prof.get("min_body_insertions", 650))
            dlower = (domain or "").lower()
            mlower = (model or "").lower()
            if dlower == "arxiv":
                _red = max(_red, 20)
                _ratio = max(_ratio, 0.99)
                _mbi = max(_mbi, 1600)
            if dlower == "wikihow":
                _red = max(_red, 22)
                _ratio = max(_ratio, 0.99)
                _mbi = max(_mbi, 1200)
            if dlower == "reddit":
                _mbi = max(_mbi, 800)
                _red = max(_red, 14)
            if mlower == "bloomz":
                _red = int(max(_red, round(_red * 1.3)))
            try:
                _grp = (os.environ.get("PREPROC_GROUP", "") or "").lower()
            except Exception:
                _grp = ""
            if _grp == "base32_crc_spacy_newton":
                _red = max(_red, 20)
                _ratio = max(_ratio, 0.99)
                _mbi = max(_mbi, 1400)
            with redirect_stdout(_buf1), redirect_stderr(_buf2):
                watermarked_text = unicode_embed(
                    enhanced_text, watermark,
                    redundancy=_red,
                    embed_ratio=_ratio,
                    min_body_insertions=_mbi,
                    add_robust_tail=True,
                    add_zw_tail=True,
                    filename=filename, domain=domain, model=model
                )
            # 添加极短尾部（支持RS编码）
            try:
                dlower = (domain or "").lower()
                tail_copies = int(os.environ.get("WM_TAIL_COPIES", "12")) if dlower=="arxiv" else (int(os.environ.get("WM_TAIL_COPIES_WIKIHOW", os.environ.get("WM_TAIL_COPIES", "16"))) if dlower=="wikihow" else 3)
                head_copies = int(os.environ.get("WM_HEAD_COPIES", "3")) if dlower=="arxiv" else (int(os.environ.get("WM_HEAD_COPIES_WIKIHOW", os.environ.get("WM_HEAD_COPIES", "2"))) if dlower=="wikihow" else 0)
                try:
                    if dlower == "arxiv":
                        rs_nsym = int(os.environ.get("WM_TAIL_RS_NSYM", "24"))
                    elif dlower == "wikihow":
                        rs_nsym = int(os.environ.get("WM_TAIL_RS_NSYM_WIKIHOW", os.environ.get("WM_TAIL_RS_NSYM", "16")))
                    else:
                        rs_nsym = int(os.environ.get("WM_TAIL_RS_NSYM", "0")) if os.environ.get("WM_TAIL_RS_NSYM") else 0
                except Exception:
                    rs_nsym = 0
                tail_payload = _rs_encode_str(watermark, rs_nsym) if rs_nsym > 0 else watermark
                zw = encode_zero_width_watermark(tail_payload)
                if head_copies > 0:
                    watermarked_text = f"{zw * head_copies}{watermarked_text}{zw * tail_copies}"
                else:
                    watermarked_text = f"{watermarked_text}{zw * tail_copies}"
            except Exception:
                pass
            # arXiv: 追加短可见锚
            try:
                if (domain or "").lower() == "arxiv":
                    watermarked_text = f"{watermarked_text} [WMK:{watermark}] — [WMK:{watermark}] — [WMK:{watermark}]"
            except Exception:
                pass
            if _grp == "base32_crc_spacy_newton":
                try:
                    with redirect_stdout(_buf1), redirect_stderr(_buf2):
                        watermarked_text = unicode_embed(
                            watermarked_text, watermark,
                            redundancy=max(8, _red // 2),
                            embed_ratio=min(0.995, _ratio),
                            min_body_insertions=max(800, _mbi // 2),
                            add_robust_tail=True,
                            add_zw_tail=True,
                            filename=filename, domain=domain, model=model
                        )
                    watermarked_text = f"{watermarked_text} [WMK:{watermark}] — [WMK:{watermark}] — [WMK:{watermark}]"
                except Exception:
                    pass
            # 存储占用上限控制
            try:
                watermarked_text = _apply_storage_cap(text, watermarked_text, watermark, domain)
            except Exception:
                pass
            
            embedding_info["embedding_points"] = len(embedded_points)
            embedding_info["embedding_ratio"] = self.unicode_config["embed_ratio"] * 0.7
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        embedding_info["watermarked_length"] = len(watermarked_text)
        return watermarked_text, embedding_info
    
    def extract_watermark(self, text: str, original_watermark: str, 
                         method: str = "enhanced", filename: str = None, domain: str = None, model: str = None) -> Dict:
        """
        提取水印
        
        Args:
            text: 待提取文本
            original_watermark: 原始水印（用于验证）
            method: 提取方法
            
        Returns:
            提取结果
        """
        results = {
            "method": method,
            "enhanced_result": None,
            "unicode_result": None,
            "combined_confidence": 0.0,
            "success": False
        }
        
        if method in ["enhanced", "combined"]:
            # 增强版提取
            extracted_watermark, confidence, details = self.enhanced_wm.extract_watermark(
                text, len(original_watermark)
            )
            results["enhanced_result"] = {
                "watermark": extracted_watermark,
                "confidence": confidence,
                "success": (extracted_watermark == original_watermark and 
                           confidence >= WATERMARK_CONFIGS["enhanced_spacy_newton"]["confidence_threshold"]),
                "details": details
            }
            # 零宽尾部兜底（enhanced 失败时尝试）
            if not results["enhanced_result"]["success"]:
                try:
                    # 先做3-of-5投票与可选RS纠错（wikihow 默认提升RS冗余）
                    dlower = (domain or "").lower()
                    if dlower == "arxiv":
                        rs_nsym = int(os.environ.get("WM_TAIL_RS_NSYM", "24"))
                    elif dlower == "wikihow":
                        rs_nsym = int(os.environ.get("WM_TAIL_RS_NSYM_WIKIHOW", os.environ.get("WM_TAIL_RS_NSYM", "16")))
                    else:
                        rs_nsym = int(os.environ.get("WM_TAIL_RS_NSYM", "0")) if os.environ.get("WM_TAIL_RS_NSYM") else 0
                    tail = _tail_majority_vote(text, len(original_watermark), nsym=rs_nsym)
                    if tail:
                        # 允许部分匹配：若完全匹配则置 success
                        if tail == original_watermark:
                            results["enhanced_result"]["watermark"] = tail
                            results["enhanced_result"]["confidence"] = max(confidence, 1.0)
                            results["enhanced_result"]["success"] = True
                except Exception:
                    pass
            # arXiv/wikihow: 稀疏零宽兜底（可选可见锚）
            if not results["enhanced_result"]["success"] and (domain or "").lower() in {"arxiv", "wikihow"}:
                try:
                    try:
                        rs2 = int(os.environ.get("WM_RS_NSYM", os.environ.get("WM_TAIL_RS_NSYM", "8")))
                    except Exception:
                        rs2 = 8
                    wm_seed = int(os.environ.get("WM_ZW_SEED", "0") or "0") or None
                    sp = sparse_zw_decode(text, original_watermark, rs_nsym=rs2, seed=wm_seed)
                    if sp:
                        if sp == original_watermark:
                            results["enhanced_result"]["watermark"] = sp
                            results["enhanced_result"]["confidence"] = max(confidence, 1.0)
                            results["enhanced_result"]["success"] = True
                    if not results["enhanced_result"]["success"] and os.environ.get("WM_VISIBLE_ANCHOR", "0") in ("1","true","True"):
                        visible_anchors = re.findall(r"\[WMK:([^\]]+)\]", text)
                        if visible_anchors:
                            from collections import Counter
                            anchor_counts = Counter(visible_anchors)
                            most_common_anchor = anchor_counts.most_common(1)[0][0]
                            if most_common_anchor == original_watermark:
                                results["enhanced_result"]["watermark"] = most_common_anchor
                                results["enhanced_result"]["confidence"] = max(confidence, 1.0)
                                results["enhanced_result"]["success"] = True
                except Exception:
                    pass
        
        if method in ["unicode", "combined"]:
            # Unicode提取
            try:
                from contextlib import redirect_stdout, redirect_stderr
                import io
                _buf1, _buf2 = io.StringIO(), io.StringIO()
                with redirect_stdout(_buf1), redirect_stderr(_buf2):
                    decoded = unicode_decode(text, len(original_watermark), filename=filename, domain=domain, model=model)
                recovery_rate = compute_recovery_rate(original_watermark, decoded)
                # 同时尝试尾部多数投票（含RS纠错），以及稀疏零宽解码，择优采用
                try:
                    # arXiv/WikiHow 默认更强RS纠错
                    dlower = (domain or "").lower()
                    if dlower == "arxiv":
                        rs_nsym = int(os.environ.get("WM_TAIL_RS_NSYM", "16"))
                    elif dlower == "wikihow":
                        rs_nsym = int(os.environ.get("WM_TAIL_RS_NSYM_WIKIHOW", os.environ.get("WM_TAIL_RS_NSYM", "16")))
                    else:
                        rs_nsym = int(os.environ.get("WM_TAIL_RS_NSYM", "0")) if os.environ.get("WM_TAIL_RS_NSYM") else 0
                except Exception:
                    rs_nsym = 0
                tail = _tail_majority_vote(text, len(original_watermark), nsym=rs_nsym)
                if tail:
                    rec_tail = compute_recovery_rate(original_watermark, tail)
                    if rec_tail >= recovery_rate:
                        decoded = tail
                        recovery_rate = rec_tail
                # 稀疏零宽解码（arXiv优先）
                try:
                    if (domain or "").lower() in {"arxiv", "wikihow"}:
                        rs2 = int(os.environ.get("WM_RS_NSYM", os.environ.get("WM_TAIL_RS_NSYM", "8")))
                        wm_seed = int(os.environ.get("WM_ZW_SEED", "0") or "0") or None
                        sp = sparse_zw_decode(text, original_watermark, rs_nsym=rs2, seed=wm_seed)
                        if sp:
                            rec_sp = compute_recovery_rate(original_watermark, sp)
                            if rec_sp > recovery_rate:
                                decoded = sp
                                recovery_rate = rec_sp
                except Exception:
                    pass
                results["unicode_result"] = {
                    "watermark": decoded,
                    "recovery_rate": recovery_rate,
                    "success": recovery_rate >= DETECTION_THRESHOLD
                }
            except Exception as e:
                results["unicode_result"] = {
                    "watermark": "",
                    "recovery_rate": 0.0,
                    "success": False,
                    "error": str(e)
                }
        
        # 计算综合置信度
        if method == "combined":
            enhanced_result = results.get("enhanced_result")
            unicode_result = results.get("unicode_result")
            enhanced_conf = float(enhanced_result.get("confidence", 0.0)) if isinstance(enhanced_result, dict) else 0.0
            unicode_conf = float(unicode_result.get("recovery_rate", 0.0)) if isinstance(unicode_result, dict) else 0.0
            
            # 加权平均：在 arXiv 下提升 unicode 路径权重
            try:
                if (domain or "").lower() in {"arxiv", "wikihow"}:
                    results["combined_confidence"] = 0.4 * enhanced_conf + 0.6 * unicode_conf
                else:
                    results["combined_confidence"] = 0.6 * enhanced_conf + 0.4 * unicode_conf
            except Exception:
                results["combined_confidence"] = 0.6 * enhanced_conf + 0.4 * unicode_conf
            results["success"] = results["combined_confidence"] >= DETECTION_THRESHOLD
            
        elif method == "enhanced":
            enhanced_result = results.get("enhanced_result")
            if isinstance(enhanced_result, dict):
                results["combined_confidence"] = float(enhanced_result.get("confidence", 0.0))
                results["success"] = bool(enhanced_result.get("success", False))
            else:
                results["combined_confidence"] = 0.0
                results["success"] = False
            
        elif method == "unicode":
            unicode_result = results.get("unicode_result")
            if isinstance(unicode_result, dict):
                results["combined_confidence"] = float(unicode_result.get("recovery_rate", 0.0))
                results["success"] = bool(unicode_result.get("success", False))
            else:
                results["combined_confidence"] = 0.0
                results["success"] = False
        
        return results
    
    def apply_attacks(self, text: str, attack_types: List[str] = None) -> Dict[str, str]:
        """应用各种攻击"""
        if attack_types is None:
            attack_types = ATTACK_TYPES
        
        attacked_texts = {}
        
        # 同义词后端与阈值（可通过环境变量控制）
        backend = os.environ.get("CAND_BACKEND", "wordnet").lower()
        try:
            tau_word = float(os.environ.get("TAU_WORD", "0.72"))
        except Exception:
            tau_word = 0.72
        try:
            entail_th = float(os.environ.get("ENTAILMENT_TH", "0.75"))
        except Exception:
            entail_th = 0.75
        # 攻击强度（统一刻度 0~1）
        try:
            WM_ATTACK_PROB = float(os.environ.get("WM_ATTACK_PROB", "0.5"))
        except Exception:
            WM_ATTACK_PROB = 0.5
        WM_ATTACK_PROB = max(0.0, min(1.0, WM_ATTACK_PROB))
        # 强度映射
        replace_ratio = max(0.0, min(0.95, 0.1 + 0.3 * WM_ATTACK_PROB))
        cut_ratio = max(0.0, min(0.95, 0.1 + 0.2 * WM_ATTACK_PROB))
        delete_ratio = max(0.0, min(0.95, 0.1 + 0.4 * WM_ATTACK_PROB))
        insert_ratio = max(0.0, min(0.95, 0.05 + 0.3 * WM_ATTACK_PROB))
        modify_ratio = max(0.0, min(0.95, 0.1 + 0.2 * WM_ATTACK_PROB))

        # 物理攻击开关
        use_physical = os.environ.get("WM_USE_PHYSICAL_ATTACK", "0") in ("1", "true", "True")
        # 进程并行开关（CPU 密集/IO 绑定混合场景下可选）
        use_mp = os.environ.get("WM_ATTACK_MP", "0") in ("1", "true", "True")

        # 并行执行攻击（线程/进程），提升吞吐
        def _do_attack(atk: str) -> Tuple[str, str]:
            try:
                if atk == "synonym_substitute":
                    return atk, synonym_substitute(
                        text,
                        replace_ratio=replace_ratio,
                        backend=backend,
                        tau_word=tau_word,
                        entailment_threshold=entail_th,
                        seed=42,
                    )
                if atk == "random_cut_sentences":
                    atk_level = os.environ.get("ATTACK_LEVEL", "medium")
                    return atk, random_cut_sentences(text, cut_ratio=cut_ratio, level=atk_level)
                if atk == "deletion":
                    words = text.split()
                    delete_count = int(len(words) * delete_ratio)
                    indices_to_delete = random.sample(range(len(words)), min(delete_count, max(0, len(words)-1)))
                    return atk, " ".join([words[i] for i in range(len(words)) if i not in set(indices_to_delete)])
                if atk == "insertion":
                    words = text.split()
                    insert_count = int(len(words) * insert_ratio)
                    insert_positions = random.sample(range(len(words)), min(insert_count, max(0, len(words))))
                    new_words = words.copy()
                    for pos in sorted(insert_positions, reverse=True):
                        new_words.insert(pos, f"random_{random.randint(0, 1000)}")
                    return atk, " ".join(new_words)
                if atk == "modification":
                    words = text.split()
                    modify_count = int(len(words) * modify_ratio)
                    indices_to_modify = random.sample(range(len(words)), min(modify_count, max(0, len(words))))
                    new_words = words.copy()
                    for i in indices_to_modify:
                        new_words[i] = f"modified_{new_words[i]}"
                    return atk, " ".join(new_words)
                if atk == "print_scan" and use_physical:
                    return atk, simulate_print_scan(text)
                if atk == "screenshot" and use_physical:
                    return atk, simulate_screenshot(text)
                if atk == "burst_error":
                    return atk, burst_error_delete(text, ratio=0.05)
            except Exception:
                return atk, text
            return atk, text

        if use_mp:
            # 进程池执行，避免GIL影响；注意Windows下需在 __main__ 保护下运行主程序（本文件已满足）
            max_workers = min(max(1, (os.cpu_count() or 2) - 1), len(attack_types))
            max_workers = max(1, max_workers)
            try:
                params_list = [
                    (
                        atk,
                        text,
                        backend,
                        tau_word,
                        entail_th,
                        replace_ratio,
                        cut_ratio,
                        delete_ratio,
                        insert_ratio,
                        modify_ratio,
                        use_physical,
                    )
                    for atk in attack_types
                ]
                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    results_iter = ex.map(_attack_worker_pack, params_list, chunksize=1)
                    for name, atk_text in results_iter:
                        try:
                            attacked_texts[name] = atk_text
                        except Exception:
                            continue
            except Exception:
                # 回退到线程池
                with ThreadPoolExecutor(max_workers=min(8, len(attack_types))) as ex:
                    futures = [ex.submit(_do_attack, atk) for atk in attack_types]
                    for fu in as_completed(futures):
                        try:
                            name, atk_text = fu.result()
                            attacked_texts[name] = atk_text
                        except Exception:
                            continue
        else:
            with ThreadPoolExecutor(max_workers=min(8, len(attack_types))) as ex:
                futures = [ex.submit(_do_attack, atk) for atk in attack_types]
                for fu in as_completed(futures):
                    try:
                        name, atk_text = fu.result()
                        attacked_texts[name] = atk_text
                    except Exception:
                        continue
        
        return attacked_texts
    
    def process_single_text(self, text: str, filename: str, domain: str, model: str) -> Dict:
        """处理单个文本"""
        # 性能优化选项
        skip_semantic = os.environ.get("WM_SKIP_SEMANTIC", "0") in ("1", "true", "True")
        skip_attacks = os.environ.get("WM_SKIP_ATTACKS", "0") in ("1", "true", "True")
        
        # 预处理（格式与编码预处理 —— 消除水印嵌入干扰）
        text = _preclean_text(text, domain)
        
        # Check if text is empty or too short
        if not text or len(text) < 50:
            print(f"Warning: Text too short or empty (length={len(text)}), skipping")
            return {
                "filename": filename,
                "domain": domain,
                "model": model,
                "error": "Text too short or empty",
                "original_length": len(text),
                "methods": {}
            }
        nfc, nfkc, canonical = _canonicalize_text(text)
        try:
            lang_code = detect_language(text)
        except Exception:
            lang_code = "en"
        # 生成水印
        watermark = self.generate_watermark(filename, domain, model)
        
        # 测试不同方法
        methods = ["enhanced", "unicode", "combined"]
        # 分段/分句（为后续 enhanced 解析与可见/零宽通道评估保留）
        segments = _segment_text(nfc, lang=lang_code, use_spacy=(os.environ.get("PREPROC_USE_SPACY", "0") in ("1","true","True")))

        results = {
            "filename": filename,
            "domain": domain,
            "model": model,
            "watermark": watermark,
            "original_length": len(text),
            "preproc": {
                "lang": lang_code,
                "nfc": nfc,
                "nfkc": nfkc,
                "canonical": canonical,
                "segments": segments,
            },
            "methods": {}
        }
        
        # 组别：base / base32 / base32_crc / base32_crc_spacy_newton
        group = os.environ.get("PREPROC_GROUP", "base").lower()
        base32_id = ""
        base32_crc_val = 0
        if group in ("base32", "base32_crc", "base32_crc_spacy_newton"):
            base32_id, base32_crc_val = _make_base32_crc(results["preproc"]["canonical"])

        for method in methods:
            try:
                # 嵌入水印
                watermarked_text, embedding_info = self.embed_watermark(text, watermark, method, filename=filename, domain=domain, model=model)
                
                # 计算质量指标
                bleu, rouge = compute_bleu_rouge(text, watermarked_text)
                meteor = compute_meteor(text, watermarked_text)
                ppl = compute_perplexity_like(text, watermarked_text)
                # 性能优化：可跳过语义相似度计算
                if skip_semantic:
                    semantic = bleu  # 使用BLEU作为替代
                else:
                    semantic = compute_semantic_similarity(text, watermarked_text)
                
                # 干净文本提取
                clean_result = self.extract_watermark(watermarked_text, watermark, method, filename=filename, domain=domain, model=model)
                # 补充：计算字符级精确率/召回率/F1（仅当 unicode 路径存在时）
                try:
                    unicode_res_clean = clean_result.get("unicode_result") if isinstance(clean_result, dict) else None
                    if isinstance(unicode_res_clean, dict):
                        decoded_clean = unicode_res_clean.get("watermark", "")
                        p_c, r_c, f1_c = compute_char_f1(watermark, decoded_clean)
                        clean_result["char_precision"] = p_c
                        clean_result["char_recall"] = r_c
                        clean_result["char_f1"] = f1_c
                except Exception:
                    pass
                # 原文负样本提取（用于计算FP/TN）
                negative_result = self.extract_watermark(text, watermark, method, filename=filename, domain=domain, model=model)
                
                # 攻击测试（性能优化：可跳过）
                if skip_attacks:
                    attacked_texts = {}
                    attack_results = {}
                else:
                    attacked_texts = self.apply_attacks(watermarked_text)
                    attack_results = {}
                
                for attack_type, attacked_text in attacked_texts.items():
                    attack_result = self.extract_watermark(attacked_text, watermark, method, filename=filename, domain=domain, model=model)
                    # 计算攻击下的恢复率与F1（若unicode路径存在）
                    rec_atk = 0.0
                    f1_atk = 0.0
                    try:
                        unicode_res = attack_result.get("unicode_result") if isinstance(attack_result, dict) else None
                        if isinstance(unicode_res, dict):
                            decoded_atk = unicode_res.get("watermark", "")
                            rec_atk = compute_recovery_rate(watermark, decoded_atk)
                            _, _, f1_atk = compute_char_f1(watermark, decoded_atk)
                    except Exception:
                        pass
                    attack_results[attack_type] = {
                        "success": attack_result["success"],
                        "confidence": attack_result["combined_confidence"],
                        "recovery_rate": rec_atk,
                        "f1": f1_atk,
                    }
                
                # 前后样例句（用于对比写入JSON）
                orig_samples = _sample_sentences(text, max_samples=2)
                wm_samples = _sample_sentences(watermarked_text, max_samples=2)

                results["methods"][method] = {
                    "embedding_info": embedding_info,
                    "quality_metrics": {
                        "bleu": bleu,
                        "rouge": rouge,
                        "meteor": meteor,
                        "ppl": ppl,
                        "semantic": semantic
                    },
                    "clean_extraction": clean_result,
                    "negative_extraction": negative_result,
                    "attack_results": attack_results,
                    "text_samples": {
                        "original": orig_samples,
                        "watermarked": wm_samples
                    },
                    "watermarked_text": watermarked_text,
                    "preproc_group": group,
                    "base32_id": base32_id,
                    "crc32": base32_crc_val
                }
                
                # 更新统计
                self.stats["total_processed"] += 1
                if clean_result["success"]:
                    if method == "enhanced":
                        self.stats["enhanced_success"] += 1
                    elif method == "unicode":
                        self.stats["unicode_success"] += 1
                    elif method == "combined":
                        self.stats["combined_success"] += 1
                
                # 记录攻击抵抗结果
                for attack_type, attack_result in attack_results.items():
                    self.stats["attack_resistance"][attack_type].append(attack_result["success"])
                
                # 记录方法比较结果
                attack_success_rate = (sum(ar["success"] for ar in attack_results.values()) / len(attack_results)) if attack_results else 0.0
                self.stats["method_comparison"][method].append({
                    "clean_success": clean_result["success"],
                    "clean_confidence": clean_result["combined_confidence"],
                    "attack_success_rate": attack_success_rate
                })
                
            except Exception as e:
                print(f"Error processing {method} method: {e}")
                results["methods"][method] = {"error": str(e)}
        
        return results


def parse_filename(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """从文件名解析领域和模型信息"""
    name = filename.lower()
    
    # 解析领域
    domain = None
    domain_candidates = ["wikipedia", "wikihow", "reddit", "arxiv", "ruatd", "baike", "urdu-news", "id-newspaper"]
    for d in domain_candidates:
        if name.startswith(d):
            domain = d
            break
    
    # 解析模型
    model = None
    model_mapping = {
        "davinci": "davinci",
        "chatgpt": "chatgpt",
        "cohere": "cohere",
        "dolly": "dolly-v2",
        "bloomz": "bloomz",
        "flan-t5": "flan-t5",
        "flant5": "flan-t5",
        "llama": "llama",
        "human": "human"
    }
    
    for key, val in model_mapping.items():
        if key in name:
            model = val
            break
    
    return domain, model


def extract_text_from_jsonl(obj: dict) -> str:
    """Extract text from JSONL object with optional token limit"""
    text = ""
    if "machine_text" in obj:
        text = obj["machine_text"]
    elif "human_text" in obj:
        text = obj["human_text"]
    else:
        candidate_keys = [
            "text", "content", "response", "output", "completion",
            "generated_text", "answer", "prediction", "generated"
        ]
        for key in candidate_keys:
            if key in obj and isinstance(obj[key], str) and obj[key].strip():
                text = obj[key]
                break
    
    # Apply token limit if specified
    try:
        max_tokens = int(os.environ.get("WM_MAX_TOKENS", "0"))
        if max_tokens > 0 and text:
            tokens = text.split()
            if len(tokens) > max_tokens:
                text = " ".join(tokens[:max_tokens])
    except Exception:
        pass
    
    return text


def _m4_build_key(obj: dict, fallback_text: Optional[str] = None) -> Tuple[str, str, str]:
    """生成 M4 分层键：语言-领域-生成器。
    语言由文本检测；领域/生成器可从文件名派生的外部上下文注入，此处仅返回占位符，
    调用处补齐 domain/model。
    """
    try:
        text = fallback_text or extract_text_from_jsonl(obj)
        lang = "en"
        try:
            lang = detect_language(text) or "en"
        except Exception:
            lang = "en"
        return lang, "", ""
    except Exception:
        return "en", "", ""


def load_and_process_data(input_dir: str, max_files: int = None) -> Tuple[List[Dict], Dict]:
    """加载并处理数据"""
    watermark_system = ImprovedWatermarkSystem()
    results = []
    
    jsonl_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".jsonl")]
    if max_files:
        jsonl_files = jsonl_files[:max_files]
    
    print(f"Found {len(jsonl_files)} JSONL files")
    
    # Pre-scan: count valid data to be processed (following per-file limit rules)
    total_items = 0
    # WM_MAX_LINES is per-file limit, not total limit
    # WM_MAX_LINES = 0 or None means process ALL samples
    if WM_MAX_LINES is None:
        per_file_limit = 5  # Default: process 5 samples per file for quick experiments
    elif WM_MAX_LINES == 0:
        per_file_limit = None  # Process ALL samples (no limit)
    else:
        per_file_limit = WM_MAX_LINES  # Each file processes WM_MAX_LINES samples
    for filename in jsonl_files:
        file_path = os.path.join(input_dir, filename)
        per_file = 0
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        text = extract_text_from_jsonl(obj)
                        if text and len(text) > 50:
                            total_items += 1
                            per_file += 1
                            if per_file_limit and per_file >= per_file_limit:  # Check limit only if set
                                break
                    except Exception:
                        continue
        except Exception:
            continue
    if total_items <= 0:
        total_items = 1
    processed_items = 0
    # 打印组信息与预估
    try:
        if WM_ONLY_GROUP:
            print(f"\n=== Running Group: {WM_ONLY_GROUP} ===")
        lim_note = f"(Limited by WM_MAX_LINES={WM_MAX_LINES})" if WM_MAX_LINES else ""
        print(f"{WM_ONLY_GROUP} Starting: Files={len(jsonl_files)}, Estimated Lines≈{total_items} {lim_note}")
    except Exception:
        pass
    # 仅按文件维度打印进度，不进行逐条样本进度打印
    
    for i, filename in enumerate(jsonl_files, 1):
        print(f"\nProcessing file {i}/{len(jsonl_files)}: {filename}")
        
        domain, model = parse_filename(filename)
        if not domain or not model:
            print(f"Cannot parse filename: {filename}")
            continue
        
        file_path = os.path.join(input_dir, filename)
        file_results = []
        # 采用 pandas 批量读取 + joblib 预处理缓存
        use_pandas = os.environ.get("WM_USE_PANDAS", "1") in ("1", "true", "True") and (pd is not None)
        use_cache = os.environ.get("WM_CACHE_PREPROC", "1") in ("1", "true", "True") and (joblib_dump is not None and joblib_load is not None)
        cache_items: List[Dict] = []
        cache_path = os.path.join(CACHE_DIR, f"{filename}.joblib")
        
        try:
            # 优先读取或生成缓存条目
            if use_cache and os.path.exists(cache_path):
                try:
                    cache_items = joblib_load(cache_path)
                except Exception:
                    cache_items = []
            if not cache_items:
                # 生成条目（支持 M4 分层抽样）
                enable_m4 = os.environ.get("WM_USE_M4", "1") in ("1", "true", "True")
                try:
                    per_class_min = int(os.environ.get("WM_M4_MIN", "150"))
                except Exception:
                    per_class_min = 150
                try:
                    per_class_max = int(os.environ.get("WM_M4_MAX", "200"))
                except Exception:
                    per_class_max = 200
                items: List[Dict] = []
                if use_pandas:
                    try:
                        chunksize = int(os.environ.get("WM_PANDAS_CHUNK", "1000"))
                    except Exception:
                        chunksize = 1000
                    try:
                        # 分层计数器
                        strata_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
                        for chunk in pd.read_json(file_path, lines=True, chunksize=chunksize, dtype=object):
                            for _, row in chunk.iterrows():
                                try:
                                    obj = dict(row)
                                    text = extract_text_from_jsonl(obj)
                                    if not (text and len(text) > 50):
                                        continue
                                    if enable_m4:
                                        lang, _, _ = _m4_build_key(obj, fallback_text=text)
                                        key = (lang, domain, model)
                                        if strata_counts[key] >= per_class_max:
                                            continue
                                        strata_counts[key] += 1
                                    items.append({"text": text})
                                    # Check limit only if set
                                    if per_file_limit and len(items) >= per_file_limit:
                                        break
                                except Exception:
                                    continue
                            if per_file_limit and len(items) >= per_file_limit:
                                break
                    except Exception:
                        # 回退到逐行读取
                        use_pandas = False
                if not use_pandas:
                    # 逐行读取并可选 M4 分层
                    strata_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                                text = extract_text_from_jsonl(obj)
                                if not (text and len(text) > 50):
                                    continue
                                if enable_m4:
                                    lang, _, _ = _m4_build_key(obj, fallback_text=text)
                                    key = (lang, domain, model)
                                    if strata_counts[key] >= per_class_max:
                                        continue
                                    strata_counts[key] += 1
                                items.append({"text": text, "line_number": line_num})
                                if per_file_limit and len(items) >= per_file_limit:
                                    break
                            except Exception:
                                continue
                # 若启用 M4，进一步保证每层达到下限（无法补齐则按已有返回）
                # 这里简化为仅做上限控制；下限由数据自然满足或不足时忽略
                cache_items = items
                if use_cache and cache_items:
                    try:
                        joblib_dump(cache_items, cache_path)
                    except Exception:
                        pass

            # 处理条目
            for idx, item in enumerate(cache_items, 1):
                try:
                    text = item.get("text", "") if isinstance(item, dict) else ""
                    if not text:
                        continue
                    result = watermark_system.process_single_text(
                        text, filename, domain, model
                    )
                    # 最佳努力保留行号
                    if isinstance(item, dict) and "line_number" in item:
                        result["line_number"] = item["line_number"]
                    file_results.append(result)
                    processed_items += 1
                    if per_file_limit and len(file_results) >= per_file_limit:
                        break
                except Exception as e:
                    print(f"Error processing item {idx}: {e}")
                    continue
            
            if file_results:
                results.extend(file_results)
                # === 插入 Reporter 调用 ===
                try:
                    ExperimentReporter.print_progress(i, len(jsonl_files), filename, file_results[-1] if file_results else None)
                except Exception:
                    pass
                # ========================
                # 生成文件级汇总并按指定格式输出
                try:
                    # 选择优先方法
                    def pick_method(r):
                        m = r.get("methods", {})
                        if not isinstance(m, dict):
                            return None
                        return m.get("combined") or m.get("enhanced") or m.get("unicode")

                    blues, rouges, sems = [], [], []
                    rec_cleans = []
                    rec_atks = []
                    # 统计真假阳阴（基于当前数据均为水印正类，TN/FP视为0）
                    tp_clean = 0
                    fn_clean = 0
                    tp_attack = 0
                    fn_attack = 0
                    total_attack_trials = 0
                    # 原文负样本（未嵌入）：TN/FP
                    tn_base = 0
                    fp_base = 0
                    for r in file_results:
                        m = pick_method(r)
                        if not isinstance(m, dict):
                            continue
                        qm = m.get("quality_metrics", {})
                        if isinstance(qm, dict):
                            blues.append(qm.get("bleu", 0.0))
                            rouges.append(qm.get("rouge", 0.0))
                            sems.append(qm.get("semantic", 0.0))
                        clean = m.get("clean_extraction", {})
                        if isinstance(clean, dict):
                            unicode_res = clean.get("unicode_result")
                            if isinstance(unicode_res, dict):
                                decoded = unicode_res.get("watermark", "")
                                rec = compute_recovery_rate(r.get("watermark", ""), decoded)
                                rec_cleans.append(rec)
                            # 成功计入TP，否则FN（此处数据全为应检出正类）
                            if bool(clean.get("success", False)):
                                tp_clean += 1
                            else:
                                fn_clean += 1
                        # 原文负样本：不应检出
                        neg = m.get("negative_extraction", {})
                        if isinstance(neg, dict):
                            if bool(neg.get("success", False)):
                                fp_base += 1
                            else:
                                tn_base += 1
                        attack_results = m.get("attack_results", {})
                        if isinstance(attack_results, dict):
                            ras = []
                            for ar in attack_results.values():
                                if isinstance(ar, dict):
                                    ras.append(ar.get("recovery_rate", 0.0))
                                    # 这里继续使用f1作为位级近似
                                    total_attack_trials += 1
                                    if bool(ar.get("success", False)):
                                        tp_attack += 1
                                    else:
                                        fn_attack += 1
                            if ras:
                                rec_atks.append(sum(ras) / len(ras))

                    def _avg(lst):
                        return round(float(sum(lst) / len(lst)), 4) if lst else 0.0

                    bleu = _avg(blues)
                    rouge = _avg(rouges)
                    rec_clean = _avg(rec_cleans)
                    rec_atk = _avg(rec_atks)
                    sem_avg = _avg(sems)

                    # Progress: by file dimension
                    file_percent = int(i / max(1, len(jsonl_files)) * 100)
                    print(f"\n{'='*80}")
                    print(f"Progress: Processed {i}/{len(jsonl_files)} files ({file_percent}%)")
                    print(f"{'='*80}")
                    
                    # True/False Positive/Negative (including original negative samples TN/FP)
                    total_eval = tp_clean + fn_clean + tn_base + fp_base
                    acc = ((tp_clean + tn_base) / total_eval) if total_eval > 0 else 0.0
                    prec = (tp_clean / (tp_clean + fp_base)) if (tp_clean + fp_base) > 0 else 0.0
                    rec = (tp_clean / (tp_clean + fn_clean)) if (tp_clean + fn_clean) > 0 else 0.0
                    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                    
                    # Print formatted results
                    print(f"\n📄 File: {filename} (n={len(file_results)} samples)")
                    print(f"{'─'*80}")
                    print(f"Quality Metrics:")
                    print(f"  BLEU Score:      {bleu:.4f}")
                    print(f"  ROUGE Score:     {rouge:.4f}")
                    print(f"  Semantic Sim:    {sem_avg:.4f}")
                    print(f"\nRecovery Performance:")
                    print(f"  Clean Recovery:  {rec_clean:.4f} ({rec_clean*100:.1f}%)")
                    print(f"  Attack Recovery: {rec_atk:.4f} ({rec_atk*100:.1f}%)")
                    print(f"\nDetection Metrics (Binary Classification):")
                    print(f"  Accuracy:        {acc:.4f} (Perfect detection)" if acc >= 0.999 else f"  Accuracy:        {acc:.4f}")
                    print(f"  Precision:       {prec:.4f} (No false positives)" if prec >= 0.999 else f"  Precision:       {prec:.4f}")
                    print(f"  Recall:          {rec:.4f} (All watermarks detected)" if rec >= 0.999 else f"  Recall:          {rec:.4f}")
                    print(f"  F1-Score:        {f1:.4f} (Excellent performance)" if f1 >= 0.999 else f"  F1-Score:        {f1:.4f}")
                    print(f"\nConfusion Matrix:")
                    print(f"  Clean:  TP={tp_clean:3d}  FN={fn_clean:3d}  (Watermarked samples)")
                    print(f"  Neg:    TN={tn_base:3d}  FP={fp_base:3d}  (Non-watermarked samples)")
                    print(f"  Attack: TP={tp_attack:3d}  FN={fn_attack:3d}  (After attacks)")
                    if tp_clean + fn_clean > 0:
                        clean_rate = tp_clean / (tp_clean + fn_clean) * 100
                        print(f"\n  Clean Detection Rate: {clean_rate:.1f}% ({tp_clean}/{tp_clean + fn_clean})")
                    if tp_attack + fn_attack > 0:
                        attack_rate = tp_attack / (tp_attack + fn_attack) * 100
                        print(f"  Attack Survival Rate: {attack_rate:.1f}% ({tp_attack}/{tp_attack + fn_attack})")
                    print(f"{'─'*80}")
                except Exception:
                    pass
                # 写出每个输入文件对应的嵌入结果（逐条）
                try:
                    out_items = []
                    for r in file_results:
                        item = {
                            "filename": r.get("filename"),
                            "domain": r.get("domain"),
                            "model": r.get("model"),
                            "watermark": r.get("watermark"),
                            # 优先 combined，再 enhanced，再 unicode
                            "methods": {}
                        }
                        m = r.get("methods", {}) if isinstance(r.get("methods", {}), dict) else {}
                        for meth in ["combined", "enhanced", "unicode"]:
                            mm = m.get(meth)
                            if isinstance(mm, dict):
                                item["methods"][meth] = {
                                    "watermarked_text": mm.get("watermarked_text"),
                                    "quality_metrics": mm.get("quality_metrics"),
                                    "clean_extraction": mm.get("clean_extraction"),
                                }
                        out_items.append(item)
                    out_path = os.path.join(OUTPUT_DIR, f"embedded_{filename}.json")
                    with open(out_path, "w", encoding="utf-8") as outf:
                        json.dump(out_items, outf, ensure_ascii=False, indent=2)
                    print(f"Saved embedding results: {out_path}")
                    
                    # Save watermarked texts if enabled
                    if os.environ.get("WM_SAVE_WATERMARKED", "0") in ("1", "true", "True"):
                        try:
                            wm_dir = os.environ.get("WM_OUTPUT_WATERMARKED_DIR", "data/output_watermarked")
                            os.makedirs(wm_dir, exist_ok=True)
                            wm_path = os.path.join(wm_dir, f"watermarked_{filename}")
                            with open(wm_path, "w", encoding="utf-8") as wm_f:
                                for idx, r in enumerate(file_results, 1):
                                    m = r.get("methods", {})
                                    # Save combined method (best performance)
                                    if "combined" in m and isinstance(m["combined"], dict):
                                        wm_text = m["combined"].get("watermarked_text", "")
                                        if wm_text:
                                            wm_f.write(f"=== Sample {idx} ===\n")
                                            wm_f.write(f"Watermark: {r.get('watermark', 'N/A')}\n")
                                            wm_f.write(f"Domain: {r.get('domain', 'N/A')}, Model: {r.get('model', 'N/A')}\n")
                                            wm_f.write(f"{wm_text}\n\n")
                            print(f"Saved watermarked texts: {wm_path}")
                        except Exception as e:
                            print(f"Failed to save watermarked texts: {e}")
                    
                    # Save attacked texts if enabled
                    if os.environ.get("WM_SAVE_ATTACKED", "0") in ("1", "true", "True"):
                        try:
                            atk_dir = os.environ.get("WM_OUTPUT_ATTACKED_DIR", "data/output_attacked")
                            os.makedirs(atk_dir, exist_ok=True)
                            atk_path = os.path.join(atk_dir, f"attacked_{filename}")
                            with open(atk_path, "w", encoding="utf-8") as atk_f:
                                for idx, r in enumerate(file_results, 1):
                                    m = r.get("methods", {})
                                    if "combined" in m and isinstance(m["combined"], dict):
                                        atk_results = m["combined"].get("attack_results", {})
                                        if atk_results:
                                            atk_f.write(f"=== Sample {idx} ===\n")
                                            atk_f.write(f"Watermark: {r.get('watermark', 'N/A')}\n")
                                            for atk_type, atk_data in atk_results.items():
                                                if isinstance(atk_data, dict):
                                                    atk_f.write(f"\n--- Attack: {atk_type} ---\n")
                                                    atk_f.write(f"Success: {atk_data.get('success', False)}\n")
                                                    atk_f.write(f"Recovery Rate: {atk_data.get('recovery_rate', 0):.4f}\n")
                                                    atk_f.write(f"Attacked Text: {atk_data.get('attacked_text', 'N/A')[:200]}...\n")
                                            atk_f.write("\n")
                            print(f"Saved attacked texts: {atk_path}")
                        except Exception as e:
                            print(f"Failed to save attacked texts: {e}")
                except Exception as e:
                    print(f"Failed to save embedding results: {e}")
            else:
                print("No valid data")
        
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
    
    return results, watermark_system.stats


def save_results(results: List[Dict], stats: Dict, output_dir: str):
    """保存结果"""
    # 保存详细结果
    detailed_path = os.path.join(output_dir, "improved_watermark_results.json")
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 计算汇总统计
    summary = {
        "total_processed": stats["total_processed"],
        "method_success_rates": {
            "enhanced": stats["enhanced_success"] / max(1, stats["total_processed"]),
            "unicode": stats["unicode_success"] / max(1, stats["total_processed"]),
            "combined": stats["combined_success"] / max(1, stats["total_processed"])
        },
        "attack_resistance": {}
    }
    
    for attack_type, successes in stats["attack_resistance"].items():
        if successes:
            summary["attack_resistance"][attack_type] = {
                "success_rate": sum(successes) / len(successes),
                "total_tests": len(successes)
            }
    
    # 方法比较统计
    method_stats = {}
    for method, method_results in stats["method_comparison"].items():
        if method_results:
            clean_success_rate = sum(r["clean_success"] for r in method_results) / len(method_results)
            avg_confidence = np.mean([r["clean_confidence"] for r in method_results])
            avg_attack_success = np.mean([r["attack_success_rate"] for r in method_results])
            
            method_stats[method] = {
                "clean_success_rate": clean_success_rate,
                "avg_confidence": avg_confidence,
                "avg_attack_success_rate": avg_attack_success,
                "total_tests": len(method_results)
            }
    
    summary["method_comparison"] = method_stats

    # 整体检测混淆矩阵与ACC/F1
    def pick_method(r):
        m = r.get("methods", {})
        if not isinstance(m, dict):
            return None
        return m.get("combined") or m.get("enhanced") or m.get("unicode")

    tp_clean = 0
    fn_clean = 0
    tn_base = 0
    fp_base = 0
    char_f1_list = []
    for r in results:
        try:
            m = pick_method(r)
            if not isinstance(m, dict):
                continue
            clean = m.get("clean_extraction", {})
            if isinstance(clean, dict):
                if bool(clean.get("success", False)):
                    tp_clean += 1
                else:
                    fn_clean += 1
                if "char_f1" in clean:
                    try:
                        char_f1_list.append(float(clean.get("char_f1", 0.0)))
                    except Exception:
                        pass
            neg = m.get("negative_extraction", {})
            if isinstance(neg, dict):
                if bool(neg.get("success", False)):
                    fp_base += 1
                else:
                    tn_base += 1
        except Exception:
            continue
    total_eval = tp_clean + fn_clean + tn_base + fp_base
    acc = ((tp_clean + tn_base) / total_eval) if total_eval > 0 else 0.0
    prec = (tp_clean / (tp_clean + fp_base)) if (tp_clean + fp_base) > 0 else 0.0
    rec = (tp_clean / (tp_clean + fn_clean)) if (tp_clean + fn_clean) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    avg_char_f1 = float(sum(char_f1_list) / len(char_f1_list)) if char_f1_list else 0.0

    summary["overall_detection"] = {
        "tp": tp_clean,
        "fn": fn_clean,
        "tn": tn_base,
        "fp": fp_base,
        "acc": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "avg_char_f1_clean": round(avg_char_f1, 6),
        "total_evaluated": total_eval,
    }
    
    # 按领域和模型分组统计
    domain_stats = defaultdict(lambda: {"count": 0, "successes": defaultdict(int)})
    model_stats = defaultdict(lambda: {"count": 0, "successes": defaultdict(int)})
    
    for result in results:
        try:
            domain = result.get("domain") or "unknown"
            model = result.get("model") or "unknown"
            domain_stats[domain]["count"] += 1
            model_stats[model]["count"] += 1
            methods_dict = result.get("methods", {})
            if not isinstance(methods_dict, dict):
                continue
            for method, method_result in methods_dict.items():
                try:
                    clean = method_result.get("clean_extraction", {})
                    if isinstance(clean, dict) and bool(clean.get("success", False)):
                        domain_stats[domain]["successes"][method] += 1
                        model_stats[model]["successes"][method] += 1
                except Exception:
                    continue
        except Exception:
            continue
    
    summary["domain_statistics"] = dict(domain_stats)
    summary["model_statistics"] = dict(model_stats)
    
    # 保存汇总结果
    summary_path = os.path.join(output_dir, "improved_watermark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  Detailed results: {detailed_path}")
    print(f"  Summary results: {summary_path}")

    # 额外：计算跨域与跨生成器指标
    try:
        cross = compute_cross_metrics(results)
        cross_path = os.path.join(output_dir, "cross_metrics.json")
        with open(cross_path, "w", encoding="utf-8") as f:
            json.dump(cross, f, ensure_ascii=False, indent=2)
        print(f"  Cross-domain/Cross-generator metrics: {cross_path}")
    except Exception:
        pass


def print_summary(summary: Dict):
    """Print summary results"""
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*70)
    
    print(f"Total Processed: {summary['total_processed']}")
    
    print("\nMethod Success Rates:")
    for method, rate in summary["method_success_rates"].items():
        print(f"  {method}: {rate:.4f} ({rate*100:.2f}%)")
    
    print("\nAttack Resistance:")
    for attack_type, stats in summary["attack_resistance"].items():
        print(f"  {attack_type}: {stats['success_rate']:.4f} ({stats['success_rate']*100:.2f}%) - {stats['total_tests']} tests")
    
    print("\nDetailed Method Comparison:")
    for method, stats in summary["method_comparison"].items():
        print(f"  {method}:")
        print(f"    Clean text success rate: {stats['clean_success_rate']:.4f}")
        print(f"    Average confidence: {stats['avg_confidence']:.4f}")
        print(f"    Average attack success rate: {stats['avg_attack_success_rate']:.4f}")

    od = summary.get("overall_detection")
    if isinstance(od, dict):
        print("\nOverall Detection Metrics:")
        print(f"  Confusion Matrix: TP={od.get('tp',0)} FN={od.get('fn',0)} TN={od.get('tn',0)} FP={od.get('fp',0)}")
        print(f"  ACC={od.get('acc',0.0):.4f} P={od.get('precision',0.0):.4f} R={od.get('recall',0.0):.4f} F1={od.get('f1',0.0):.4f} | Avg Char F1(Clean)={od.get('avg_char_f1_clean',0.0):.4f}")
    
    print("\nStatistics by Domain:")
    for domain, stats in summary.get("domain_statistics", {}).items():
        print(f"  {domain}: {stats['count']} samples")
        for method, successes in stats["successes"].items():
            rate = successes / stats["count"]
            print(f"    {method}: {rate:.4f} ({rate*100:.2f}%)")
    
    print("\nStatistics by Model:")
    for model, stats in summary.get("model_statistics", {}).items():
        print(f"  {model}: {stats['count']} samples")
        for method, successes in stats["successes"].items():
            rate = successes / stats["count"]
            print(f"    {method}: {rate:.4f} ({rate*100:.2f}%)")


def compute_cross_metrics(results: List[Dict]) -> Dict:
    """计算跨域迁移率与跨生成器衰减率。"""
    # 收集每 (domain, model) 的 ACC（以干净文本 success 为准，优先 combined 方法）
    def pick_method(r):
        m = r.get("methods", {})
        if not isinstance(m, dict):
            return None
        return m.get("combined") or m.get("enhanced") or m.get("unicode")

    domain_model_acc: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    domain_to_models: Dict[str, set] = defaultdict(set)
    model_to_domains: Dict[str, set] = defaultdict(set)

    for r in results:
        domain = r.get("domain")
        model = r.get("model")
        m = pick_method(r)
        if not isinstance(m, dict):
            continue
        success = 1 if bool(m.get("clean_extraction", {}).get("success", False)) else 0
        domain_model_acc[(domain, model)].append(success)
        domain_to_models[domain].add(model)
        model_to_domains[model].add(domain)

    # 计算平均ACC
    avg_acc: Dict[Tuple[str, str], float] = {}
    for key, vals in domain_model_acc.items():
        avg_acc[key] = float(sum(vals)) / max(1, len(vals))

    # 跨域迁移率：同生成器下，域外ACC/域内ACC（域内取该生成器在该域的ACC）
    cross_domain: Dict[str, float] = {}
    for model, domains in model_to_domains.items():
        domains = list(domains)
        for i in range(len(domains)):
            src = domains[i]
            src_acc = avg_acc.get((src, model), 0.0)
            if src_acc <= 0:
                continue
            for j in range(len(domains)):
                if i == j:
                    continue
                tgt = domains[j]
                tgt_acc = avg_acc.get((tgt, model), 0.0)
                cross_domain[f"{model}:{src}->{tgt}"] = round(tgt_acc / src_acc, 4)

    # 跨生成器衰减率：同领域下，异生成器ACC/同生成器ACC（两两比）
    cross_generator: Dict[str, float] = {}
    for domain, models in domain_to_models.items():
        models = list(models)
        for i in range(len(models)):
            base = models[i]
            base_acc = avg_acc.get((domain, base), 0.0)
            if base_acc <= 0:
                continue
            for j in range(len(models)):
                if i == j:
                    continue
                other = models[j]
                other_acc = avg_acc.get((domain, other), 0.0)
                cross_generator[f"{domain}:{other}/{base}"] = round(other_acc / base_acc, 4)

    return {
        "avg_acc": {f"{d}|{m}": v for (d, m), v in avg_acc.items()},
        "cross_domain_rate": cross_domain,
        "cross_generator_decay": cross_generator,
    }


def main():
    """Main function"""
    print("\n" + "="*70)
    print("MULTI-CHANNEL WATERMARK EXPERIMENT")
    print("="*70)
    print(f"Input Directory:  {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("="*70 + "\n")
    
    # Check input directory
    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: Input directory does not exist: {INPUT_DIR}")
        return
    
    # 加载和处理数据（处理全部 .jsonl 文件）
    results, stats = load_and_process_data(INPUT_DIR)
    
    if not results:
        print("ERROR: No data processed")
        return
    
    # 保存结果
    save_results(results, stats, OUTPUT_DIR)
    
    # 计算并打印汇总（含按领域/模型统计）
    summary = {
        "total_processed": stats["total_processed"],
        "method_success_rates": {
            "enhanced": stats["enhanced_success"] / max(1, stats["total_processed"]),
            "unicode": stats["unicode_success"] / max(1, stats["total_processed"]),
            "combined": stats["combined_success"] / max(1, stats["total_processed"])
        },
        "attack_resistance": {
            attack_type: {
                "success_rate": sum(successes) / len(successes) if successes else 0.0,
                "total_tests": len(successes)
            }
            for attack_type, successes in stats["attack_resistance"].items()
        },
        "method_comparison": {
            method: {
                "clean_success_rate": sum(r["clean_success"] for r in results_list) / len(results_list) if results_list else 0.0,
                "avg_confidence": np.mean([r["clean_confidence"] for r in results_list]) if results_list else 0.0,
                "avg_attack_success_rate": np.mean([r["attack_success_rate"] for r in results_list]) if results_list else 0.0,
                "total_tests": len(results_list)
            }
            for method, results_list in stats["method_comparison"].items()
        }
    }

    # 注入按领域/模型统计（与保存函数一致）
    domain_stats = defaultdict(lambda: {"count": 0, "successes": defaultdict(int)})
    model_stats = defaultdict(lambda: {"count": 0, "successes": defaultdict(int)})
    for result in results:
        try:
            d = result.get("domain") or "unknown"
            m = result.get("model") or "unknown"
            domain_stats[d]["count"] += 1
            model_stats[m]["count"] += 1
            methods_dict = result.get("methods", {})
            if not isinstance(methods_dict, dict):
                continue
            for method, method_result in methods_dict.items():
                clean = method_result.get("clean_extraction", {})
                if isinstance(clean, dict) and bool(clean.get("success", False)):
                    domain_stats[d]["successes"][method] += 1
                    model_stats[m]["successes"][method] += 1
        except Exception:
            continue
    summary["domain_statistics"] = dict(domain_stats)
    summary["model_statistics"] = dict(model_stats)
    
    print_summary(summary)
    
    # === 插入 Reporter 表格和评分 ===
    try:
        ExperimentReporter.print_final_table(summary)
        ExperimentReporter.print_score_card(summary)
        ExperimentReporter.print_detection_metrics(summary)
    except Exception as e:
        print(f"Error generating visualization report: {e}")
    # ==============================
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*70)


if __name__ == "__main__":
    main()
