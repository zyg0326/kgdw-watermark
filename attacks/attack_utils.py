import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import random
try:
    import pytesseract  # 可选：OCR
    from PIL import Image, ImageFilter, ImageDraw, ImageFont
    import numpy as _np
except Exception:
    pytesseract = None
    Image = None
    ImageFilter = None
    ImageDraw = None
    ImageFont = None
    _np = None
try:
    from methods.candidate_selector import CandidateSelector
except Exception:
    CandidateSelector = None

# 自动下载所有必要的NLTK资源
def download_nltk_resources():
    required_resources = [
        'averaged_perceptron_tagger_eng',
        'punkt',
        'wordnet'
    ]
    for resource in required_resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            print(f"正在下载NLTK资源: {resource}")
            nltk.download(resource, quiet=True)

# 初始化时自动下载缺失的资源
download_nltk_resources()

def synonym_substitute(text, replace_ratio=0.3, backend: str = "wordnet", tau_word: float = 0.72, entailment_threshold: float = 0.75, seed: int = 42):
    """同义词替换攻击（可插拔后端）

    backend: "wordnet" | "sbert" | "nli_embed"
    tau_word/entailment_threshold 仅在 sbert/nli_embed 时生效
    """
    words = word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    new_words = []
    replace_count = int(len(words) * replace_ratio)
    replaced = 0

    selector = None
    if CandidateSelector is not None:
        try:
            selector = CandidateSelector(backend=backend, tau_word=tau_word, entailment_threshold=entailment_threshold, seed=seed)
        except Exception:
            selector = None

    import random as _rnd
    _rnd.seed(seed)

    for word, pos in pos_tags:
        if len(word) < 3:
            new_words.append(word)
            continue
        if replaced >= replace_count:
            new_words.append(word)
            continue

        candidates = []
        if selector is not None:
            try:
                candidates = selector.select_candidates(word, pos=pos, context=text, top_k=6)
            except Exception:
                candidates = []

        if not candidates:
            # 回退：直接用 WordNet
            synonyms = []
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.name() != word:
                        synonyms.append(lemma.name())
            candidates = [s.replace('_', ' ') for s in synonyms]

        # 选一个候选，尽量避免形态不自然
        chosen = None
        for cand in candidates:
            if cand and cand.isascii() and cand.lower() != word.lower():
                chosen = cand
                break
        if chosen is None and candidates:
            chosen = candidates[0]

        if chosen:
            new_words.append(chosen)
            replaced += 1
        else:
            new_words.append(word)

    return ' '.join(new_words)

def random_cut_sentences(text, cut_ratio=0.2, level: str = "medium"):
    """随机切割句子攻击（支持强度分层）"""
    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return text
    
    # 按强度分层
    level = (level or "medium").lower()
    factor = {"low": 0.5, "medium": 1.0, "high": 1.5}.get(level, 1.0)
    cut_count = max(1, int(len(sentences) * cut_ratio * factor))
    import random
    cut_indices = random.sample(range(len(sentences)), cut_count)
    
    remaining_sentences = [
        sent for i, sent in enumerate(sentences) 
        if i not in cut_indices
    ]
    
    return ' '.join(remaining_sentences)


def light_synonym_nouns(text: str, noun_ratio: float = 0.025) -> str:
    """
    轻量级近义替换，仅针对名词，默认替换约2.5%的词，尽量不改变可读性。
    """
    if not text or noun_ratio <= 0:
        return text
    words = word_tokenize(text)
    try:
        pos_tags = nltk.pos_tag(words)
    except Exception:
        return text
    noun_indices = [i for i, (_, pos) in enumerate(pos_tags) if pos in ("NN", "NNS", "NNP", "NNPS")]
    if not noun_indices:
        return text
    import random as _rnd
    _rnd.seed(42)
    k = max(1, int(len(words) * noun_ratio))
    choose = set(_rnd.sample(noun_indices, min(k, len(noun_indices))))
    new_words = []
    for idx, (w, pos) in enumerate(pos_tags):
        if idx in choose and len(w) >= 3:
            syns = []
            for syn in wordnet.synsets(w, pos=wordnet.NOUN):
                for lemma in syn.lemmas():
                    cand = lemma.name().replace('_', ' ')
                    if cand.lower() != w.lower() and cand.isascii():
                        syns.append(cand)
            if syns:
                new_words.append(syns[0])
                continue
        new_words.append(w)
    return ' '.join(new_words)


# ===== 新增：物理/数字攻击 =====
def simulate_print_scan(text: str, dpi: int = 600, target_dpi: int = 300) -> str:
    """打印-扫描-识别攻击（OCR）。依赖 pillow+pytesseract，不可用时返回原文。"""
    if Image is None or pytesseract is None or ImageDraw is None or ImageFont is None or _np is None:
        return text
    try:
        font = None
        try:
            font = ImageFont.truetype("Arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        dummy = Image.new('L', (1, 1), 255)
        d0 = ImageDraw.Draw(dummy)
        bbox = d0.textbbox((0, 0), text, font=font)
        img = Image.new('L', (max(32, bbox[2] + 20), max(32, bbox[3] + 20)), 255)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), text, font=font, fill=0)
        # 下采样与模糊
        w = int(img.width * target_dpi / max(1, dpi))
        h = int(img.height * target_dpi / max(1, dpi))
        img = img.resize((max(1, w), max(1, h)), Image.Resampling.LANCZOS)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
        # 椒盐噪声
        arr = _np.array(img)
        noise = (_np.random.rand(*arr.shape) < 0.02).astype('uint8') * 60
        arr = _np.clip(arr - noise, 0, 255).astype('uint8')
        img = Image.fromarray(arr)
        # OCR
        lang = 'eng'
        try:
            from methods.localized_unicode import detect_language as _dl
            lc = _dl(text)
            if lc == 'ar_ur':
                lang = 'ara'
            elif lc == 'bg':
                lang = 'bul'
            elif lc == 'zh':
                lang = 'chi_sim'
        except Exception:
            pass
        return pytesseract.image_to_string(img, lang=lang).strip()
    except Exception:
        return text


def simulate_screenshot(text: str, original_res=(1920, 1080), target_res=(1280, 720)) -> str:
    """截图-下采样-识别攻击（OCR）。依赖 pillow+pytesseract，不可用时返回原文。"""
    if Image is None or pytesseract is None or ImageDraw is None or ImageFont is None:
        return text
    try:
        font = None
        try:
            font = ImageFont.truetype("Arial.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
        img = Image.new('L', original_res, 255)
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), text, font=font, fill=0)
        img = img.resize(target_res, Image.Resampling.BILINEAR)
        lang = 'eng'
        try:
            from methods.localized_unicode import detect_language as _dl
            if _dl(text) == 'ar_ur':
                lang = 'ara'
        except Exception:
            pass
        return pytesseract.image_to_string(img, lang=lang).strip()
    except Exception:
        return text


def burst_error_delete(text: str, ratio: float = 0.05) -> str:
    """突发错误：连续删除约 ratio 的字符。"""
    if not text:
        return text
    try:
        n = len(text)
        k = max(1, int(n * max(0.0, min(0.95, ratio))))
        if k >= n:
            return ''
        start = random.randint(0, n - k)
        return text[:start] + text[start + k:]
    except Exception:
        return text
    