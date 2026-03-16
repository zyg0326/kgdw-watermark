#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强版spaCy+牛顿插值水印系统
借鉴拉格朗日插值的MCP投票机制，提高水印恢复率和抗攻击能力

主要改进：
1. 保持spaCy依存分析确定嵌入位置
2. 使用牛顿插值进行位置预测和恢复
3. 借鉴拉格朗日插值的MCP投票机制
4. 多路径投票聚合提高恢复准确性
5. 增强的抗攻击能力
"""

import numpy as np
import re
import hashlib
import random
import os
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
import spacy
from dataclasses import dataclass

# 导入现有的牛顿插值函数
from methods.localized_unicode import (
    newton_interpolation, _adaptive_newton_predict, _segment_indices,
    get_dependency_stability_score, extract_candidate_positions,
    ENCODE_CHARS, ROBUST_ENCODE_CHARS, SEPARATORS, UNICODE_CHARS
)

# 加载spaCy模型
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("警告: 无法加载spaCy模型，将使用简化版本")
    nlp = None


@dataclass
class WatermarkPoint:
    """水印点数据结构"""
    position: int
    character: str
    confidence: float
    source: str  # 'spacy', 'newton', 'manual'
    context: Dict = None


@dataclass
class WatermarkLine:
    """水印直线数据结构"""
    points: List[WatermarkPoint]
    slope: float
    intercept: float
    confidence: float
    point_count: int


class EnhancedSpacyNewtonWatermark:
    """增强版spaCy+牛顿插值水印系统"""
    
    def __init__(self, field_size: int = 8, secret_key: str = "enhanced_key"):
        """
        初始化增强版水印系统
        
        Args:
            field_size: 有限域大小（用于MCP算法）
            secret_key: 密钥
        """
        self.field_size = field_size
        self.field_size_power = 2 ** field_size
        self.secret_key = secret_key
        
        # 水印配置
        self.min_collinear_points = 3
        self.confidence_threshold = 0.7
        self.consensus_threshold = 0.6
        
        # 字符集
        self.encode_chars = ENCODE_CHARS
        self.robust_chars = ROBUST_ENCODE_CHARS
        self.separators = SEPARATORS
        # 线性映射模数（使用编码字符表长度，便于将 y 映射为字符）
        self.char_mod = max(2, len(self.encode_chars))
        # 多项式运算模数（素数，近似域算术；避免除法失败）
        self.poly_mod = 104729
    
    def _hash_to_field(self, data: str) -> int:
        """将数据哈希到有限域"""
        hash_obj = hashlib.sha256((self.secret_key + data).encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        return hash_int % self.field_size_power
    
    def _generate_position_hash(self, text: str, position: int) -> int:
        """根据文本和位置生成哈希值"""
        context = text[max(0, position-10):position+10]
        data = f"{context}_{position}_{self.secret_key}"
        return self._hash_to_field(data)
    
    def extract_enhanced_positions(self, text: str, num_points: int = 20) -> List[WatermarkPoint]:
        """
        使用增强的spaCy+牛顿插值方法提取候选位置
        
        Args:
            text: 输入文本
            num_points: 需要的点数
            
        Returns:
            水印点列表
        """
        points = []
        
        if nlp is None:
            # 简化版本：均匀分布
            return self._extract_uniform_positions(text, num_points)
        
        # 1. spaCy依存分析位置 + 句内二次均匀化（优先）
        spacy_points = self._extract_spacy_positions(text, num_points * 2)
        # 句内均匀 + 中位±偏移，并避让数学/LaTeX上下文
        try:
            sent_uniform = self._extract_sentence_uniform_positions(text, num_points)
            spacy_points.extend(sent_uniform)
        except Exception:
            pass
        points.extend(spacy_points)

        # 2. 基于牛顿插值对依存候选进行筛选与生成稳定位点
        filtered_points = self._select_positions_with_newton(
            text=text,
            spacy_points=spacy_points,
            num_points=num_points,
            window=384,
            overlap=48,
            min_gap=4,
        )
        if filtered_points and len(filtered_points) >= int(0.7 * num_points):
            return filtered_points[:num_points]

        # 3. 回退：增加插值导向候选+MCP优化
        newton_points = self._extract_newton_positions(text, num_points, spacy_points)
        points.extend(newton_points)
        optimized_points = self._optimize_positions_with_mcp(text, points, num_points)
        return optimized_points[:num_points]

    def _is_math_context(self, text: str, pos: int, window: int = 24) -> bool:
        try:
            l = max(0, pos - window)
            r = min(len(text), pos + window)
            ctx = text[l:r]
            if ('$' in ctx) or ('\\begin' in ctx) or ('\\end' in ctx):
                return True
            for sym in ['∑', '±', '×', '=', '^', '_']:
                if sym in ctx:
                    return True
            return False
        except Exception:
            return False

    def _extract_sentence_uniform_positions(self, text: str, num_points: int) -> List[WatermarkPoint]:
        points: List[WatermarkPoint] = []
        if nlp is None:
            return points
        try:
            doc = nlp(text)
            sents = list(doc.sents)
            if not sents:
                return points
            total_len = sum(max(1, s.end_char - s.start_char) for s in sents)
            if total_len <= 0:
                return points
            # 按句长比例分配配额
            alloc = []
            for s in sents:
                sl = max(1, s.end_char - s.start_char)
                cnt = max(1, int(round(num_points * (sl / total_len))))
                alloc.append(cnt)
            # 调整配额和为 num_points
            diff = sum(alloc) - num_points
            i = 0
            while diff != 0 and len(alloc) > 0:
                if diff > 0 and alloc[i] > 1:
                    alloc[i] -= 1
                    diff -= 1
                elif diff < 0:
                    alloc[i] += 1
                    diff += 1
                i = (i + 1) % len(alloc)
            # 句内均匀 + 中位±偏移（5–15%句长），避开数学上下文
            rng = random.Random(42)
            for si, s in enumerate(sents):
                start = s.start_char
                end = s.end_char
                slen = max(1, end - start)
                k = max(1, alloc[si])
                for t in range(k):
                    frac = (t + 0.5) / k
                    base = start + int(frac * slen)
                    jitter = int(rng.uniform(0.05, 0.15) * slen)
                    pos = base + (jitter if (t % 2) == 0 else -jitter)
                    pos = min(max(pos, start), end - 1)
                    if 0 <= pos < len(text) and not self._is_math_context(text, pos):
                        ch = text[pos]
                        points.append(WatermarkPoint(position=pos, character=ch, confidence=0.6, source='sent_uniform', context={'sent_index': si}))
            return points[: num_points * 2]
        except Exception:
            return points
    
    def _extract_spacy_positions(self, text: str, num_points: int) -> List[WatermarkPoint]:
        """使用spaCy提取位置"""
        doc = nlp(text)
        points = []
        
        for token in doc:
            score = get_dependency_stability_score(token)
            if score > 0.5:
                # 在词的不同位置创建点
                positions = [
                    token.idx,
                    token.idx + len(token.text) // 2,
                    token.idx + len(token.text)
                ]
                
                for pos in positions:
                    if 0 <= pos < len(text):
                        char = text[pos] if pos < len(text) else ' '
                        point = WatermarkPoint(
                            position=pos,
                            character=char,
                            confidence=score,
                            source='spacy',
                            context={'token': token.text, 'pos': token.pos_, 'dep': token.dep_}
                        )
                        points.append(point)
        
        return points
    
    def _extract_newton_positions(self, text: str, num_points: int, 
                                existing_points: List[WatermarkPoint]) -> List[WatermarkPoint]:
        """使用牛顿插值预测位置"""
        if len(existing_points) < 3:
            return []
        
        # 使用现有的牛顿插值函数
        positions = extract_candidate_positions(text, num_points * 2)
        points = []
        
        for pos in positions:
            if pos < len(text):
                char = text[pos]
                # 计算置信度（基于与现有点的距离）
                min_distance = min(abs(pos - p.position) for p in existing_points)
                confidence = max(0.1, 1.0 - min_distance / len(text))
                
                point = WatermarkPoint(
                    position=pos,
                    character=char,
                    confidence=confidence,
                    source='newton',
                    context={'predicted': True}
                )
                points.append(point)
        
        return points
    
    def _optimize_positions_with_mcp(self, text: str, points: List[WatermarkPoint], 
                                   num_points: int) -> List[WatermarkPoint]:
        """
        使用MCP思想优化位置选择
        
        借鉴拉格朗日插值的MCP算法思想：
        1. 将位置映射到有限域
        2. 寻找"共线"的位置模式
        3. 选择最稳定的位置组合
        """
        if len(points) < 3:
            return points
        
        # 将位置映射到有限域
        field_points = []
        for point in points:
            x = self._generate_position_hash(text, point.position)
            y = point.position % self.field_size_power
            field_points.append((x, y, point))
        
        # 寻找"共线"的位置模式
        stable_combinations = self._find_stable_position_combinations(field_points)
        
        # 选择最佳组合
        if stable_combinations:
            best_combination = max(stable_combinations, key=lambda x: x['stability'])
            return [p for _, _, p in best_combination['points']]
        
        # 回退到置信度排序
        return sorted(points, key=lambda p: p.confidence, reverse=True)[:num_points]
    
    def _find_stable_position_combinations(self, field_points: List[Tuple[int, int, WatermarkPoint]]) -> List[Dict]:
        """寻找稳定的位置组合"""
        combinations = []
        
        # 尝试不同的点组合
        for i in range(len(field_points)):
            for j in range(i + 1, len(field_points)):
                for k in range(j + 1, len(field_points)):
                    p1, p2, p3 = field_points[i], field_points[j], field_points[k]
                    
                    # 检查是否"共线"（在有限域中）
                    if self._are_collinear_in_field(p1[:2], p2[:2], p3[:2]):
                        stability = self._calculate_combination_stability([p1, p2, p3])
                        combinations.append({
                            'points': [p1, p2, p3],
                            'stability': stability
                        })
        
        return combinations
    
    def _are_collinear_in_field(self, p1: Tuple[int, int], p2: Tuple[int, int], 
                              p3: Tuple[int, int]) -> bool:
        """检查三点在有限域中是否共线"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        
        # 计算斜率（在有限域中）
        if x1 == x2:
            return x2 == x3  # 垂直线
        
        # 使用有限域运算
        delta_y = (y2 - y1) % self.field_size_power
        delta_x = (x2 - x1) % self.field_size_power
        
        if delta_x == 0:
            return False
        
        # 计算期望的y3
        expected_y3 = (y1 + (delta_y * (x3 - x1) // delta_x)) % self.field_size_power
        
        return expected_y3 == y3
    
    def _calculate_combination_stability(self, points: List[Tuple[int, int, WatermarkPoint]]) -> float:
        """计算位置组合的稳定性"""
        if not points:
            return 0.0
        
        # 基于置信度的稳定性
        confidence_score = np.mean([p[2].confidence for p in points])
        
        # 基于分布的稳定性
        positions = [p[2].position for p in points]
        position_variance = np.var(positions) if len(positions) > 1 else 0
        distribution_score = 1.0 / (1.0 + position_variance / 1000)
        
        # 基于来源多样性的稳定性
        sources = [p[2].source for p in points]
        source_diversity = len(set(sources)) / len(sources)
        
        return 0.4 * confidence_score + 0.3 * distribution_score + 0.3 * source_diversity
    
    def _extract_uniform_positions(self, text: str, num_points: int) -> List[WatermarkPoint]:
        """简化版本：均匀分布位置"""
        points = []
        step = max(1, len(text) // num_points)
        
        for i in range(num_points):
            pos = min(i * step, len(text) - 1)
            char = text[pos] if pos < len(text) else ' '
            point = WatermarkPoint(
                position=pos,
                character=char,
                confidence=0.5,
                source='uniform',
                context={'uniform': True}
            )
            points.append(point)
        
        return points
    
    def embed_watermark(self, text: str, watermark: str, 
                       num_points: int = 20) -> Tuple[str, List[WatermarkPoint]]:
        """
        嵌入水印
        
        Args:
            text: 原始文本
            watermark: 水印内容
            num_points: 嵌入点数
            
        Returns:
            (水印文本, 嵌入点列表)
        """
        # 组别控制：当 PREPROC_GROUP=base32_crc_spacy_newton 时，启用线性约束嵌入
        import os as _os
        group = (_os.environ.get("PREPROC_GROUP", "") or "").lower()
        if group == "base32_crc_spacy_newton":
            return self._embed_newton_segmented(text, watermark, num_points)

        # 提取候选位置（常规）
        candidate_points = self.extract_enhanced_positions(text, num_points)
        
        # 编码水印
        encoded_watermark = self._encode_watermark(watermark)
        
        # 嵌入水印
        watermarked_text = text
        embedded_points = []
        
        for i, point in enumerate(candidate_points):
            if i < len(encoded_watermark):
                char = encoded_watermark[i]
                pos = point.position
                
                # 在位置pos插入字符
                watermarked_text = watermarked_text[:pos] + char + watermarked_text[pos:]
                
                # 更新后续位置
                for j in range(i + 1, len(candidate_points)):
                    candidate_points[j].position += 1
                
                embedded_points.append(WatermarkPoint(
                    position=pos,
                    character=char,
                    confidence=point.confidence,
                    source=point.source,
                    context=point.context
                ))
        
        return watermarked_text, embedded_points

    # ---------- 线性约束嵌入与提取（MCP/MCPP思想） ----------
    def _gf_inv(self, a: int, mod: int) -> Optional[int]:
        try:
            # Python3.8+ 支持 pow(a, -1, mod)；若失败则返回None
            return pow(a, -1, mod)
        except Exception:
            # 扩展欧几里得算法
            t, newt, r, newr = 0, 1, mod, a % mod
            while newr != 0:
                q = r // newr
                t, newt = newt, t - q * newt
                r, newr = newr, r - q * newr
            if r > 1:
                return None
            if t < 0:
                t = t + mod
            return t

    def _char_index(self, c: str) -> int:
        try:
            return self.encode_chars.index(c) % self.char_mod
        except ValueError:
            # 不在编码集时，映射到0
            return 0

    def _index_char(self, i: int) -> str:
        return self.encode_chars[i % self.char_mod]

    def _derive_x_from_context(self, text: str, position: int) -> int:
        # 使用“去水印字符”的清洗上下文，降低零宽/锚对 x 的扰动
        try:
            # 清除零宽/分隔符/稳健锚可见标记
            removal = set(self.encode_chars + self.separators)
            # 追加一些零宽/控制字符
            removal.update(['\u200b','\u200c','\u200d','\u2060','\ufeff'])
            # 构建清洗文本与位置映射
            kept_chars = []
            map_idx = []  # 原始位置 -> 清洗后索引
            cleaned_index = 0
            for i, ch in enumerate(text):
                if ch in removal:
                    map_idx.append(cleaned_index)
                    continue
                kept_chars.append(ch)
                map_idx.append(cleaned_index)
                cleaned_index += 1
            # 定位到清洗后上下文窗口
            pos_clean = map_idx[min(max(0, position), len(map_idx)-1)] if map_idx else 0
            cleaned = ''.join(kept_chars)
            context = cleaned[max(0, pos_clean-8):pos_clean]
            return self._hash_to_field(context)
        except Exception:
            context = text[max(0, position-8):position]
            return self._hash_to_field(context)

    def _derive_line_params(self, watermark: str) -> Tuple[int, int]:
        # 根据密钥与水印派生斜率与截距（模 char_mod）
        import hashlib as _hl
        h = _hl.sha256((self.secret_key + "|" + watermark).encode()).digest()
        a = (h[0] + 1) % self.char_mod  # 避免0斜率退化
        b = (h[1]) % self.char_mod
        if a == 0:
            a = 1
        return a, b

    def _embed_line_constrained(self, text: str, watermark: str, num_points: int) -> Tuple[str, List[WatermarkPoint]]:
        # 先取较多候选点，便于筛选
        candidates = self.extract_enhanced_positions(text, num_points * 4)
        if not candidates:
            return self.embed_watermark(text, watermark, num_points)
        a, b = self._derive_line_params(watermark)
        watermarked_text = text
        embedded_points: List[WatermarkPoint] = []
        used_positions: Set[int] = set()
        # 逐字符寻找满足 y == a*x+b (mod M) 的点
        for ch in watermark:
            y = self._char_index(ch)
            chosen: Optional[WatermarkPoint] = None
            for p in candidates:
                if p.position in used_positions:
                    continue
                x = self._derive_x_from_context(watermarked_text, p.position)
                if (a * (x % self.char_mod) + b) % self.char_mod == y:
                    chosen = p
                    break
            if chosen is None:
                # 兜底：放宽约束使用最高置信点
                for p in candidates:
                    if p.position not in used_positions:
                        chosen = p
                        y = self._char_index(ch)
                        break
            if chosen is None:
                continue
            # 按映射字符（确保与y一致）
            mapped_char = self._index_char(y)
            pos = chosen.position
            watermarked_text = watermarked_text[:pos] + mapped_char + watermarked_text[pos:]
            # 位移后续候选位置
            for q in candidates:
                if q.position >= pos:
                    q.position += 1
            used_positions.add(pos)
            embedded_points.append(WatermarkPoint(position=pos, character=mapped_char, confidence=chosen.confidence, source="line", context={"x": int(x), "y": int(y)}))
            if len(embedded_points) >= num_points:
                break
        return watermarked_text, embedded_points

    def _mcp_line_extraction(self, text: str, expected_length: int) -> Dict:
        # 收集候选点 (x, y)
        pts: List[Tuple[int,int,int]] = []  # (x, y, pos)
        for i, ch in enumerate(text):
            if ch in self.encode_chars:
                # 预计算与缓存 x
                x = self._derive_x_from_context(text, i)
                y = self._char_index(ch)
                pts.append((x, y, i))
        if len(pts) < 3:
            return {"watermark": "", "confidence": 0.0, "method": "mcp_line_insufficient", "details": {}}
        # 限点：截断与采样
        try:
            max_n = int(os.environ.get("NEWTON_MCP_MAX_N", "300"))
        except Exception:
            max_n = 300
        if len(pts) > max_n:
            # Top-K 置信 + 间隔采样
            pts.sort(key=lambda p: p[2])
            stride = max(1, len(pts) // max_n)
            pts = pts[::stride][:max_n]
        mod = self.char_mod
        best_count = 0
        best_a, best_b = 0, 0
        # 早停阈值
        stop_th = max(0.5, min(0.95, float(os.environ.get("NEWTON_MCP_STOP_FRAC", "0.7"))))
        target = int(stop_th * len(pts))
        # O(N^2) 线搜索（带早停）
        for i in range(len(pts)):
            x1, y1, _ = pts[i]
            for j in range(i+1, len(pts)):
                x2, y2, _ = pts[j]
                dx = (x2 - x1) % mod
                dy = (y2 - y1) % mod
                inv = self._gf_inv(dx, mod)
                if inv is None:
                    continue
                a = (dy * inv) % mod
                b = (y1 - a * (x1 % mod)) % mod
                # 计数
                cnt = 0
                for (xk, yk, _) in pts:
                    if (a * (xk % mod) + b) % mod == (yk % mod):
                        cnt += 1
                if cnt > best_count:
                    best_count, best_a, best_b = cnt, a, b
                    if best_count >= target:
                        # 早停
                        seq_chars: List[str] = []
                        seq_pts = sorted([p for p in pts if (best_a * (p[0] % mod) + best_b) % mod == (p[1] % mod)], key=lambda t: t[2])
                        for _, y, _ in seq_pts[:expected_length]:
                            seq_chars.append(self._index_char(y))
                        conf = min(1.0, best_count / max(1, len(pts)))
                        return {"watermark": "".join(seq_chars), "confidence": conf, "method": "mcp_line", "details": {"support": best_count, "total": len(pts), "a": best_a, "b": best_b, "early_stop": True}}
        conf = min(1.0, best_count / max(1, len(pts)))
        # 基于最佳线提取字符序列（按位置排序）
        seq_chars: List[str] = []
        seq_pts = sorted([p for p in pts if (best_a * (p[0] % mod) + best_b) % mod == (p[1] % mod)], key=lambda t: t[2])
        for _, y, _ in seq_pts[:expected_length]:
            seq_chars.append(self._index_char(y))
        return {"watermark": "".join(seq_chars), "confidence": conf, "method": "mcp_line", "details": {"support": best_count, "total": len(pts), "a": best_a, "b": best_b, "early_stop": False}}

    # ---------- 牛顿插值：分段插值（拉格朗日等价） ----------
    def _newton_divided_diffs(self, xs: List[int], ys: List[int], mod: int) -> List[int]:
        n = len(xs)
        a = ys[:]  # 将第一列作为初始系数
        for j in range(1, n):
            for i in range(n-1, j-1, -1):
                denom = (xs[i] - xs[i-j]) % mod
                inv = self._gf_inv(denom, mod)
                if inv is None:
                    inv = 0
                a[i] = ((a[i] - a[i-1]) % mod) * inv % mod
        return a  # a[0..n-1]

    def _newton_eval(self, xs: List[int], coeffs: List[int], xq: int, mod: int) -> int:
        n = len(coeffs)
        if n == 0:
            return 0
        val = coeffs[-1]
        for i in range(n-2, -1, -1):
            val = (val * ((xq - xs[i]) % mod) + coeffs[i]) % mod
        return val % mod

    def _embed_newton_segmented(self, text: str, watermark: str, num_points: int) -> Tuple[str, List[WatermarkPoint]]:
        t = max(2, int(os.environ.get("NEWTON_T", "4")))  # 每段点数
        need = max(len(watermark), num_points)
        cands = self.extract_enhanced_positions(text, need * 2)
        if not cands:
            return self.embed_watermark(text, watermark, num_points)
        # 取按位置排序的前 L 个点
        cands.sort(key=lambda p: p.position)
        L = min(len(watermark), len(cands))
        chosen = cands[:L]
        xs = []
        ys = []
        for i, p in enumerate(chosen):
            x = self._derive_x_from_context(text, p.position) % self.poly_mod
            y = self._char_index(watermark[i]) % self.poly_mod
            xs.append(x)
            ys.append(y)
        # 分段计算系数（每段 t 个点），但嵌入值仍使用原 y（从而精确编码 watermark）
        watermarked = text
        embedded: List[WatermarkPoint] = []
        for i, p in enumerate(chosen):
            ch = watermark[i]
            mapped = self._index_char(self._char_index(ch))
            pos = p.position
            watermarked = watermarked[:pos] + mapped + watermarked[pos:]
            for q in chosen:
                if q.position >= pos:
                    q.position += 1
            embedded.append(WatermarkPoint(position=pos, character=mapped, confidence=p.confidence, source="newton", context={"segment": i // t}))
        return watermarked, embedded

    def _extract_newton_segmented(self, text: str, expected_length: int) -> Tuple[str, float, Dict]:
        t = max(2, int(os.environ.get("NEWTON_T", "4")))
        # 收集候选 (pos, x, y)
        pts: List[Tuple[int,int,int]] = []
        for i, ch in enumerate(text):
            if ch in self.encode_chars:
                x = self._derive_x_from_context(text, i) % self.poly_mod
                y = self._char_index(ch) % self.poly_mod
                pts.append((i, x, y))
        if not pts:
            return "", 0.0, {"method": "newton_segmented", "error": "no_points"}
        pts.sort(key=lambda z: z[0])
        pts = pts[:expected_length]
        # 分段拟合 + 预测
        recovered: List[str] = []
        support = 0
        for s in range(0, len(pts), t):
            seg = pts[s:s+t]
            xs = [x for _, x, _ in seg]
            ys = [y for _, _, y in seg]
            if len(xs) >= 2:
                coeffs = self._newton_divided_diffs(xs, ys, self.poly_mod)
                for (pos, x, y_obs) in seg:
                    y_pred = self._newton_eval(xs, coeffs, x, self.poly_mod)
                    recovered.append(self._index_char(y_pred % self.char_mod))
                    if y_pred % self.poly_mod == y_obs % self.poly_mod:
                        support += 1
            else:
                # 直接沿用观测
                for (_, _, y_obs) in seg:
                    recovered.append(self._index_char(y_obs % self.char_mod))
        conf = min(1.0, support / max(1, len(pts)))
        return "".join(recovered)[:expected_length], conf, {"method": "newton_segmented", "support": support, "total": len(pts)}

    def _select_positions_with_newton(self, text: str, spacy_points: List[WatermarkPoint], num_points: int, window: int = 384, overlap: int = 48, min_gap: int = 4) -> List[WatermarkPoint]:
        """在依存句法候选基础上，使用牛顿插值进行分段筛选与稳定位点生成。

        1) 将文本按 window/overlap 分段；
        2) 每段内对候选点做自适应牛顿插值，计算 rmse/曲率/覆盖度 得分；
        3) 基于插值预测生成若干稳定位置，聚合去重并按 min_gap 过滤；
        4) 产出 top-K 位点，若不足则回退由依存候选补齐。
        """
        try:
            if not text or not spacy_points:
                return []
            # 依存候选位置（去重排序）
            cand_pos = sorted(list(dict.fromkeys([max(0, min(len(text) - 1, p.position)) for p in spacy_points])))
            if len(cand_pos) < 3:
                return []

            # 分段
            segs = _segment_indices(len(text), window=window, overlap=overlap)
            pos_to_score: Dict[int, float] = {}

            for (s, e) in segs:
                local = [p for p in cand_pos if s <= p < e]
                seg_len = max(1, e - s)
                if len(local) < 3:
                    continue
                # 构造段内索引域点对 (i, s_i)
                xs_idx = list(range(len(local)))
                pts_pairs = list(zip(xs_idx, [float(p) for p in local]))
                # 自适应牛顿插值预测：目标若干预测点
                target_num = 12 if seg_len >= 256 else 8
                xs = np.linspace(0, max(1, len(local) - 1), num=target_num)
                preds, meta = _adaptive_newton_predict(xs.tolist(), pts_pairs, max_order=8)
                # 质量指标
                rmse = float(max(1e-6, meta.get("rmse", 1.0)))
                # 二阶差分近似曲率
                curv = 0.0
                if isinstance(preds, (list, tuple)) and len(preds) >= 3:
                    diffs = [preds[i + 1] - preds[i] for i in range(len(preds) - 1)]
                    sdiffs = [abs(diffs[i + 1] - diffs[i]) for i in range(len(diffs) - 1)]
                    curv = float(np.mean(sdiffs)) / max(1.0, float(seg_len))
                coverage = min(1.0, len(local) / float(seg_len))
                # 得分：α*(1/(1+rmse)) + β*coverage - γ*curv
                alpha, beta, gamma = 0.6, 0.4, 0.1
                base_score = alpha * (1.0 / (1.0 + rmse)) + beta * coverage - gamma * curv
                # 生成预测位置并累计得分
                for y in preds:
                    pos = int(max(s, min(e - 1, round(y))))
                    if 0 <= pos < len(text):
                        pos_to_score[pos] = max(pos_to_score.get(pos, 0.0), base_score)

            if not pos_to_score:
                return []

            # 过滤：字符类别与 min_gap 去重
            def _is_acceptable_char(ch: str) -> bool:
                if not ch:
                    return False
                # 避免纯空白/标点，优先字母数字
                return ch.isalnum()

            scored = sorted([(p, sc) for p, sc in pos_to_score.items() if _is_acceptable_char(text[p])], key=lambda x: x[1], reverse=True)
            selected: List[int] = []
            for p, _ in scored:
                if not selected:
                    selected.append(p)
                else:
                    if min(abs(p - q) for q in selected) >= max(1, int(min_gap)):
                        selected.append(p)
                if len(selected) >= num_points:
                    break

            # 若不足，回退用依存候选补齐（按原置信度降序）
            if len(selected) < num_points:
                remain = []
                for pt in sorted(spacy_points, key=lambda t: float(getattr(t, 'confidence', 0.5)), reverse=True):
                    p = int(max(0, min(len(text) - 1, pt.position)))
                    if (_is_acceptable_char(text[p]) and all(abs(p - q) >= max(1, int(min_gap)) for q in selected)):
                        remain.append(p)
                    if len(selected) + len(remain) >= num_points:
                        break
                selected.extend(remain[:max(0, num_points - len(selected))])

            # 构造 WatermarkPoint
            result: List[WatermarkPoint] = []
            for p in selected[:num_points]:
                c = text[p]
                # 将得分映射为置信度（0~1）
                sc = float(pos_to_score.get(p, 0.5))
                conf = float(max(0.0, min(1.0, sc)))
                result.append(WatermarkPoint(position=p, character=c, confidence=conf, source='newton_filter', context={'min_gap': min_gap}))
            return result
        except Exception:
            return []
    
    def _encode_watermark(self, watermark: str) -> str:
        """编码水印为字符序列"""
        # 使用现有的编码方法
        encoded_chars = []
        for char in watermark:
            if char in self.encode_chars:
                encoded_chars.append(char)
            else:
                # 映射到可用字符
                idx = ord(char) % len(self.encode_chars)
                encoded_chars.append(self.encode_chars[idx])
        
        return ''.join(encoded_chars)
    
    def extract_watermark(self, text: str, expected_length: int) -> Tuple[str, float, Dict]:
        """
        提取水印
        
        Args:
            text: 待提取文本
            expected_length: 期望的水印长度
            
        Returns:
            (提取的水印, 置信度, 详细信息)
        """
        # 组别触发：优先尝试线性MCP提取
        import os as _os
        group = (_os.environ.get("PREPROC_GROUP", "") or "").lower()
        if group == "base32_crc_spacy_newton":
            # 先用牛顿分段恢复，若配置禁用MCP或已恢复非空，直接返回
            newton_wm, newton_conf, newton_info = self._extract_newton_segmented(text, expected_length)
            disable_mcp = (_os.environ.get("NEWTON_DISABLE_MCP", "0") in ("1","true","True"))
            if newton_wm or disable_mcp:
                return newton_wm, newton_conf, newton_info
            mcp = self._mcp_line_extraction(text, expected_length)
            if mcp.get('watermark'):
                return mcp['watermark'], max(0.0, float(mcp.get('confidence', 0.0))), mcp

        # 提取所有可能的嵌入点
        candidate_points = self._extract_all_watermark_points(text)
        
        if len(candidate_points) < 3:
            return "", 0.0, {"error": "insufficient_points"}
        
        # 使用MCP投票机制
        mcp_result = self._mcp_voting_extraction(candidate_points, expected_length)
        
        return mcp_result['watermark'], mcp_result['confidence'], mcp_result
    
    def _extract_all_watermark_points(self, text: str) -> List[WatermarkPoint]:
        """提取所有可能的水印点"""
        points = []
        
        for i, char in enumerate(text):
            if char in self.encode_chars or char in self.separators:
                # 计算位置哈希
                position_hash = self._generate_position_hash(text, i)
                
                point = WatermarkPoint(
                    position=i,
                    character=char,
                    confidence=0.5,  # 默认置信度
                    source='extracted',
                    context={'hash': position_hash}
                )
                points.append(point)
        
        return points
    
    def _mcp_voting_extraction(self, points: List[WatermarkPoint], 
                             expected_length: int) -> Dict:
        """
        使用MCP投票机制提取水印
        
        借鉴拉格朗日插值的MCP算法：
        1. 将点映射到有限域
        2. 寻找共线点模式
        3. 投票聚合结果
        """
        if len(points) < 3:
            return {
                'watermark': '',
                'confidence': 0.0,
                'method': 'insufficient_points',
                'details': {}
            }
        
        # 将点映射到有限域
        field_points = []
        for point in points:
            # 使用文本上下文生成哈希
            context_text = f"pos_{point.position}_char_{point.character}"
            x = self._generate_position_hash(context_text, point.position)
            y = point.position % self.field_size_power
            field_points.append((x, y, point))
        
        # 寻找共线点模式
        collinear_groups = self._find_collinear_groups(field_points)
        
        if not collinear_groups:
            return {
                'watermark': '',
                'confidence': 0.0,
                'method': 'no_collinear_groups',
                'details': {}
            }
        
        # 投票聚合
        voting_result = self._aggregate_votes(collinear_groups, expected_length)
        
        return voting_result
    
    def _find_collinear_groups(self, field_points: List[Tuple[int, int, WatermarkPoint]]) -> List[List[WatermarkPoint]]:
        """寻找共线点组"""
        groups = []
        used_points = set()
        
        for i in range(len(field_points)):
            if i in used_points:
                continue
            
            # 以点i为参考点
            reference_point = field_points[i]
            collinear_points = [reference_point[2]]  # 包含参考点
            
            # 寻找与参考点共线的其他点
            for j in range(i + 1, len(field_points)):
                if j in used_points:
                    continue
                
                other_point = field_points[j]
                
                # 检查是否共线
                if self._are_collinear_in_field(reference_point[:2], other_point[:2], (0, 0)):
                    collinear_points.append(other_point[2])
                    used_points.add(j)
            
            if len(collinear_points) >= self.min_collinear_points:
                groups.append(collinear_points)
                used_points.add(i)
        
        return groups
    
    def _aggregate_votes(self, groups: List[List[WatermarkPoint]], 
                        expected_length: int) -> Dict:
        """投票聚合"""
        if not groups:
            return {
                'watermark': '',
                'confidence': 0.0,
                'method': 'no_groups',
                'details': {}
            }
        
        # 计算每个组的置信度
        group_scores = []
        for group in groups:
            confidence = np.mean([p.confidence for p in group])
            size_score = len(group) / expected_length
            diversity_score = len(set(p.source for p in group)) / len(group)
            
            total_score = 0.4 * confidence + 0.4 * size_score + 0.2 * diversity_score
            group_scores.append((group, total_score))
        
        # 选择最佳组
        best_group, best_score = max(group_scores, key=lambda x: x[1])
        
        # 从最佳组提取水印
        watermark_chars = [p.character for p in best_group]
        watermark = ''.join(watermark_chars)[:expected_length]
        
        # 计算最终置信度
        final_confidence = min(1.0, best_score * 1.2)  # 稍微提升置信度
        
        return {
            'watermark': watermark,
            'confidence': final_confidence,
            'method': 'mcp_voting',
            'details': {
                'best_group_size': len(best_group),
                'total_groups': len(groups),
                'group_scores': [score for _, score in group_scores]
            }
        }
    
    def simulate_attacks(self, text: str, watermark: str) -> Dict:
        """模拟攻击测试"""
        # 嵌入水印
        watermarked_text, embedded_points = self.embed_watermark(text, watermark)
        
        # 定义攻击
        attacks = {
            'deletion_10': self._delete_text(watermarked_text, 0.1),
            'deletion_30': self._delete_text(watermarked_text, 0.3),
            'insertion_10': self._insert_text(watermarked_text, 0.1),
            'modification_10': self._modify_text(watermarked_text, 0.1),
        }
        
        results = {}
        for attack_name, attacked_text in attacks.items():
            extracted_watermark, confidence, details = self.extract_watermark(
                attacked_text, len(watermark)
            )
            
            success = (extracted_watermark == watermark and 
                      confidence >= self.confidence_threshold)
            
            results[attack_name] = {
                'success': success,
                'confidence': confidence,
                'extracted': extracted_watermark,
                'details': details
            }
        
        return results
    
    def _delete_text(self, text: str, ratio: float) -> str:
        """删除文本攻击"""
        words = text.split()
        delete_count = int(len(words) * ratio)
        indices_to_delete = random.sample(range(len(words)), delete_count)
        return ' '.join([words[i] for i in range(len(words)) if i not in indices_to_delete])
    
    def _insert_text(self, text: str, ratio: float) -> str:
        """插入文本攻击"""
        words = text.split()
        insert_count = int(len(words) * ratio)
        insert_positions = random.sample(range(len(words)), insert_count)
        new_words = words.copy()
        for pos in sorted(insert_positions, reverse=True):
            new_words.insert(pos, f"random_{random.randint(0, 1000)}")
        return ' '.join(new_words)
    
    def _modify_text(self, text: str, ratio: float) -> str:
        """修改文本攻击"""
        words = text.split()
        modify_count = int(len(words) * ratio)
        indices_to_modify = random.sample(range(len(words)), modify_count)
        new_words = words.copy()
        for i in indices_to_modify:
            new_words[i] = f"modified_{new_words[i]}"
        return ' '.join(new_words)


# 测试函数
def test_enhanced_spacy_newton():
    """测试增强版spaCy+牛顿插值水印"""
    print("=== 增强版spaCy+牛顿插值水印测试 ===")
    
    # 创建水印系统
    wm = EnhancedSpacyNewtonWatermark(field_size=8, secret_key="test_key")
    
    # 测试文本
    test_text = "This is a comprehensive test document for evaluating the robustness of our enhanced watermarking scheme. It contains multiple sentences with varying complexity to thoroughly test the system's capabilities."
    watermark = "TEST_WM_2024"
    
    print(f"原始文本长度: {len(test_text)}")
    print(f"水印: {watermark}")
    
    # 嵌入水印
    watermarked_text, embedded_points = wm.embed_watermark(test_text, watermark)
    print(f"水印文本长度: {len(watermarked_text)}")
    print(f"嵌入点数: {len(embedded_points)}")
    
    # 提取水印
    extracted_watermark, confidence, details = wm.extract_watermark(watermarked_text, len(watermark))
    print(f"提取水印: {extracted_watermark}")
    print(f"置信度: {confidence:.4f}")
    print(f"提取方法: {details.get('method', 'unknown')}")
    
    # 攻击测试
    print("\n=== 攻击测试 ===")
    attack_results = wm.simulate_attacks(test_text, watermark)
    
    for attack_name, result in attack_results.items():
        print(f"{attack_name}: 成功={result['success']}, 置信度={result['confidence']:.4f}")
    
    return wm, test_text, watermark


if __name__ == "__main__":
    test_enhanced_spacy_newton()
