"""
可插拔词级候选选择器

设计目标：
- 统一接口：给定原词与其上下文，返回候选替换词列表。
- 默认实现：WordNet 同义词（无外部依赖）。
- 轻量占位：余弦相似（SBERT 可选）与蕴含（NLI）留接口，后续可接入 transformers。

核心API：
- CandidateSelector.select_candidates(token: str, pos: str | None, context: str | None, k: int) -> List[str]

后端切换：
- backend="wordnet"（默认）
- backend="sbert"（需要 sentence-transformers，可选）
- backend="nli_embed"（需要 transformers，可选；语义蕴含 + 向量阈值）

参数：
- tau_word (float): 词向量相似阈值（当使用 sbert/nli_embed 时生效）
- entailment_threshold (float): NLI 蕴含概率阈值（nli_embed 时生效）
"""

from typing import List, Optional
import os
import importlib
def _choose_device_for_torch() -> str:
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
            if os.environ.get("WM_FORCE_GPU") in ("1", "true", "True"):
                return "cuda"
            return "cpu"
        return "cuda"
    except Exception:
        return "cpu"



class _LRUCache:
    def __init__(self, capacity: int = 4096):
        self.capacity = int(capacity)
        self._data = {}
        self._order = []

    def get(self, key):
        if key in self._data:
            try:
                self._order.remove(key)
            except ValueError:
                pass
            self._order.append(key)
            return self._data[key]
        return None

    def set(self, key, value):
        if key in self._data:
            self._data[key] = value
            try:
                self._order.remove(key)
            except ValueError:
                pass
            self._order.append(key)
            return
        if len(self._order) >= self.capacity:
            oldest = self._order.pop(0)
            self._data.pop(oldest, None)
        self._data[key] = value
        self._order.append(key)


class CandidateSelector:
    def __init__(
        self,
        backend: str = "wordnet",
        tau_word: float = 0.72,
        entailment_threshold: float = 0.75,
        language: str = "en",
        seed: int = 42,
    ) -> None:
        self.backend = backend
        self.tau_word = float(tau_word)
        self.entailment_threshold = float(entailment_threshold)
        self.language = language
        self.seed = int(seed)

        self._wordnet_ready = False
        self._sbert_ready = False
        self._nli_ready = False
        self._cache = _LRUCache(capacity=4096)
        # 全局缓存（跨实例共享）
        global _GLOBAL_MODELS
        try:
            _GLOBAL_MODELS
        except NameError:
            _GLOBAL_MODELS = {
                "sbert_model": None,
                "sbert_name": None,
                "nli_tok": None,
                "nli_model": None,
                "nli_name": None,
                "nli_device": None,
            }

        # 延迟加载依赖，保持无依赖可运行
        if backend == "wordnet":
            self._init_wordnet()
        elif backend == "sbert":
            self._init_sbert()
        elif backend == "nli_embed":
            self._init_sbert()
            self._init_nli()
        else:
            # 回退到 wordnet
            self.backend = "wordnet"
            self._init_wordnet()

    # ---------- backends init ----------
    def _init_wordnet(self) -> None:
        try:
            importlib.import_module("nltk")
            importlib.import_module("nltk.corpus")
            importlib.import_module("nltk.tokenize")
            self._wordnet_ready = True
        except (ImportError, Exception):
            self._wordnet_ready = False

    def _init_sbert(self) -> None:
        try:
            importlib.import_module("sentence_transformers")
            self._sbert_ready = True
        except (ImportError, Exception):
            self._sbert_ready = False

    def _init_nli(self) -> None:
        try:
            importlib.import_module("transformers")
            self._nli_ready = True
        except (ImportError, Exception):
            self._nli_ready = False

    # ---------- model getters (global cache) ----------
    def _get_sbert(self):
        if not self._sbert_ready:
            return None
        from sentence_transformers import SentenceTransformer
        device = _choose_device_for_torch()
        name = os.environ.get("SBERT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        if _GLOBAL_MODELS["sbert_model"] is not None and _GLOBAL_MODELS["sbert_name"] == name:
            return _GLOBAL_MODELS["sbert_model"]
        try:
            model = SentenceTransformer(name, device=device)
            _GLOBAL_MODELS["sbert_model"] = model
            _GLOBAL_MODELS["sbert_name"] = name
            return model
        except Exception:
            return None

    def _get_nli(self):
        if not self._nli_ready:
            return None, None, None
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        name = os.environ.get("NLI_MODEL_NAME", "microsoft/deberta-v3-base-mnli")
        if _GLOBAL_MODELS["nli_tok"] is not None and _GLOBAL_MODELS["nli_model"] is not None and _GLOBAL_MODELS["nli_name"] == name:
            return _GLOBAL_MODELS["nli_tok"], _GLOBAL_MODELS["nli_model"], _GLOBAL_MODELS["nli_device"]
        try:
            tok = AutoTokenizer.from_pretrained(name)
            clf = AutoModelForSequenceClassification.from_pretrained(name)
            clf.eval()
            device = torch.device(_choose_device_for_torch())
            clf.to(device)
            _GLOBAL_MODELS["nli_tok"] = tok
            _GLOBAL_MODELS["nli_model"] = clf
            _GLOBAL_MODELS["nli_name"] = name
            _GLOBAL_MODELS["nli_device"] = device
            return tok, clf, device
        except Exception:
            return None, None, None

    # ---------- public API ----------
    def select_candidates(
        self,
        token: str,
        pos: Optional[str] = None,
        context: Optional[str] = None,
        top_k: int = 8,
    ) -> List[str]:
        if not token or len(token) < 3:
            return []
        cache_key = (self.backend, token.lower(), pos, (context[:64] if context else None), top_k, self.tau_word, self.entailment_threshold)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return list(cached)

        backend = self.backend
        if backend == "wordnet" or not (self._sbert_ready or self._nli_ready):
            out = self._select_wordnet(token, pos, top_k)
        elif backend == "sbert":
            out = self._select_sbert(token, pos, context, top_k)
        elif backend == "nli_embed":
            out = self._select_nli_embed(token, pos, context, top_k)
        else:
            out = self._select_wordnet(token, pos, top_k)
        self._cache.set(cache_key, list(out))
        return out

    # ---------- backend impls ----------
    def _select_wordnet(self, token: str, pos: Optional[str], top_k: int) -> List[str]:
        if not self._wordnet_ready:
            return []
        try:
            from nltk.corpus import wordnet as wn
        except Exception:
            return []
        pos_map = {
            "NN": wn.NOUN, "NNS": wn.NOUN, "NNP": wn.NOUN, "NNPS": wn.NOUN,
            "VB": wn.VERB, "VBD": wn.VERB, "VBG": wn.VERB, "VBN": wn.VERB, "VBP": wn.VERB, "VBZ": wn.VERB,
            "JJ": wn.ADJ, "JJR": wn.ADJ, "JJS": wn.ADJ,
            "RB": wn.ADV, "RBR": wn.ADV, "RBS": wn.ADV,
        }
        wn_pos = pos_map.get(pos, None)
        seen = set()
        cands: List[str] = []
        try:
            synsets = wn.synsets(token, pos=wn_pos) if wn_pos else wn.synsets(token)
            for syn in synsets:
                for lemma in syn.lemmas():
                    cand = lemma.name().replace("_", " ")
                    if cand.lower() != token.lower() and cand.isascii() and cand not in seen:
                        seen.add(cand)
                        cands.append(cand)
                        if len(cands) >= top_k:
                            return cands
        except Exception:
            return cands
        return cands

    def _select_sbert(self, token: str, pos: Optional[str], context: Optional[str], top_k: int) -> List[str]:
        # 轻量占位：复用 WordNet 候选 + SBERT 打分过滤（若可用，否则退化为 WordNet）
        base = self._select_wordnet(token, pos, top_k * 5)  # 先尽量多取
        if not base:
            return []
        if not self._sbert_ready:
            return base[:top_k]
        try:
            import numpy as np
            model = self._get_sbert()
            if model is None:
                return base[:top_k]
            # 词性一致性过滤（若可用）
            try:
                from nltk.corpus import wordnet as wn
                pos_map = {
                    "NN": wn.NOUN, "NNS": wn.NOUN, "NNP": wn.NOUN, "NNPS": wn.NOUN,
                    "VB": wn.VERB, "VBD": wn.VERB, "VBG": wn.VERB, "VBN": wn.VERB, "VBP": wn.VERB, "VBZ": wn.VERB,
                    "JJ": wn.ADJ, "JJR": wn.ADJ, "JJS": wn.ADJ,
                    "RB": wn.ADV, "RBR": wn.ADV, "RBS": wn.ADV,
                }
                wn_pos = pos_map.get(pos, None)
            except Exception:
                wn_pos = None
            anchor = context if context else token
            embs = model.encode([anchor] + base, normalize_embeddings=True, convert_to_numpy=True)
            q = embs[0]
            sims = np.dot(embs[1:], q)
            pairs = []
            for c, s in zip(base, sims):
                # 词性一致性：若指定了POS且候选在该POS下无同义集，则丢弃
                if wn_pos is not None:
                    try:
                        if not wn.synsets(c, pos=wn_pos):
                            continue
                    except Exception:
                        pass
                # 词形还原约束：尽量与原词形态保持一致（简单启发）
                try:
                    if pos and pos.startswith("NN") and (c.endswith("ing") or c.endswith("ed")):
                        continue
                    if pos and pos.startswith("VB") and (c.endswith("s") and not token.endswith("s")):
                        continue
                except Exception:
                    pass
                pairs.append((c, float(s)))
            pairs = [p for p in pairs if p[1] >= self.tau_word]
            pairs.sort(key=lambda x: x[1], reverse=True)
            return [c for c, _ in pairs[:top_k]]
        except Exception:
            return base[:top_k]

    def _select_nli_embed(self, token: str, pos: Optional[str], context: Optional[str], top_k: int) -> List[str]:
        # 强筛选（占位版）：SBERT 相似度 >= tau_word + （可选）NLI 蕴含 >= entailment_threshold
        cands = self._select_sbert(token, pos, context, top_k * 2)
        if not cands:
            return []
        if not self._nli_ready:
            return cands[:top_k]
        try:
            tok, clf, device = self._get_nli()
            if tok is None or clf is None:
                return cands[:top_k]
            kept: List[str] = []
            premise = context if context else token
            for cand in cands:
                import torch
                with torch.no_grad():
                    inputs = tok(premise, cand, return_tensors="pt", truncation=True, max_length=128).to(device)
                    logits = clf(**inputs).logits[0]
                    probs = torch.softmax(logits, dim=-1)
                    entail_prob = float(probs[2])  # MNLI: 0=contra,1=neutral,2=entail
                if entail_prob >= self.entailment_threshold:
                    kept.append(cand)
                if len(kept) >= top_k:
                    break
            return kept[:top_k] if kept else cands[:top_k]
        except Exception:
            return cands[:top_k]


