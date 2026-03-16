#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical Stealthiness Testing
Anti-Steganalysis Evaluation Module

Tests whether watermarked text can be distinguished from original text
using statistical analysis (PPL, entropy, word frequency, etc.)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from collections import Counter
import re
from typing import List, Dict, Tuple

class StatisticalStealthinessTest:
    """
    Test statistical detectability of watermarks
    Goal: Classifier accuracy should be ~0.5 (random guessing)
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def extract_statistical_features(self, text: str) -> np.ndarray:
        """
        Extract statistical features for steganalysis
        
        Features:
        1. Word frequency distribution (entropy)
        2. Character frequency distribution
        3. Average word length
        4. Sentence length variance
        5. Punctuation density
        6. Vocabulary richness (Type-Token Ratio)
        7. Function word ratio
        """
        features = []
        
        # Tokenize
        words = re.findall(r'\b\w+\b', text.lower())
        chars = list(text.lower())
        sentences = re.split(r'[.!?]+', text)
        
        if len(words) == 0:
            return np.zeros(20)
        
        # 1. Word frequency entropy
        word_freq = Counter(words)
        word_probs = np.array(list(word_freq.values())) / len(words)
        word_entropy = -np.sum(word_probs * np.log2(word_probs + 1e-10))
        features.append(word_entropy)
        
        # 2. Character frequency entropy
        char_freq = Counter(chars)
        char_probs = np.array(list(char_freq.values())) / len(chars)
        char_entropy = -np.sum(char_probs * np.log2(char_probs + 1e-10))
        features.append(char_entropy)
        
        # 3. Average word length
        avg_word_len = np.mean([len(w) for w in words])
        features.append(avg_word_len)
        
        # 4. Word length variance
        word_len_var = np.var([len(w) for w in words])
        features.append(word_len_var)
        
        # 5. Sentence length variance
        sent_lens = [len(s.split()) for s in sentences if s.strip()]
        sent_len_var = np.var(sent_lens) if sent_lens else 0
        features.append(sent_len_var)
        
        # 6. Punctuation density
        punct_count = len(re.findall(r'[.,!?;:]', text))
        punct_density = punct_count / len(text)
        features.append(punct_density)
        
        # 7. Type-Token Ratio (vocabulary richness)
        ttr = len(set(words)) / len(words)
        features.append(ttr)
        
        # 8. Function word ratio
        function_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                         'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were'}
        func_word_count = sum(1 for w in words if w in function_words)
        func_word_ratio = func_word_count / len(words)
        features.append(func_word_ratio)
        
        # 9-13. Top 5 most common word frequencies
        top_5_freqs = [freq for _, freq in word_freq.most_common(5)]
        while len(top_5_freqs) < 5:
            top_5_freqs.append(0)
        features.extend([f / len(words) for f in top_5_freqs])
        
        # 14-18. Character n-gram features (bigrams)
        bigrams = [text[i:i+2] for i in range(len(text)-1)]
        bigram_freq = Counter(bigrams)
        top_5_bigrams = [freq for _, freq in bigram_freq.most_common(5)]
        while len(top_5_bigrams) < 5:
            top_5_bigrams.append(0)
        features.extend([f / len(bigrams) for f in top_5_bigrams])
        
        # 19. Average sentence length
        avg_sent_len = np.mean(sent_lens) if sent_lens else 0
        features.append(avg_sent_len)
        
        # 20. Lexical diversity (unique words / total words)
        lexical_div = len(set(words)) / len(words)
        features.append(lexical_div)
        
        return np.array(features)
    
    def test_detectability(self, 
                          original_texts: List[str], 
                          watermarked_texts: List[str],
                          verbose: bool = True) -> Dict:
        """
        Test if watermarked texts can be distinguished from originals
        
        Args:
            original_texts: List of original texts
            watermarked_texts: List of watermarked texts (same length)
            verbose: Print detailed results
        
        Returns:
            Dictionary with test results
        """
        assert len(original_texts) == len(watermarked_texts), \
            "Must have equal number of original and watermarked texts"
        
        n_samples = len(original_texts)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Statistical Stealthiness Test")
            print(f"{'='*70}")
            print(f"Samples: {n_samples} pairs")
        
        # Extract features
        if verbose:
            print("Extracting statistical features...")
        
        X = []
        y = []
        
        for text in original_texts:
            features = self.extract_statistical_features(text)
            X.append(features)
            y.append(0)  # Original
        
        for text in watermarked_texts:
            features = self.extract_statistical_features(text)
            X.append(features)
            y.append(1)  # Watermarked
        
        X = np.array(X)
        y = np.array(y)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Test with multiple classifiers
        classifiers = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "SVM (Linear)": SVC(kernel='linear', random_state=42),
            "SVM (RBF)": SVC(kernel='rbf', random_state=42)
        }
        
        results = {}
        
        for clf_name, clf in classifiers.items():
            # 5-fold cross-validation
            scores = cross_val_score(clf, X_scaled, y, cv=min(5, n_samples//2), 
                                    scoring='accuracy')
            
            mean_acc = np.mean(scores)
            std_acc = np.std(scores)
            
            results[clf_name] = {
                "accuracy": mean_acc,
                "std": std_acc,
                "scores": scores
            }
            
            if verbose:
                print(f"\n{clf_name}:")
                print(f"  Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
                print(f"  Individual folds: {scores}")
                
                # Interpret results
                if mean_acc < 0.55:
                    verdict = "✓ EXCELLENT - Statistically undetectable"
                elif mean_acc < 0.65:
                    verdict = "○ GOOD - Low detectability"
                elif mean_acc < 0.75:
                    verdict = "△ MODERATE - Some detectability"
                else:
                    verdict = "✗ POOR - Easily detectable"
                
                print(f"  Verdict: {verdict}")
        
        # Overall assessment
        avg_accuracy = np.mean([r["accuracy"] for r in results.values()])
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Overall Assessment:")
            print(f"  Average Accuracy: {avg_accuracy:.4f}")
            print(f"  Baseline (Random): 0.5000")
            print(f"  Difference: {avg_accuracy - 0.5:.4f}")
            
            if avg_accuracy < 0.55:
                print(f"  ✓ PASS - Watermark is statistically stealthy")
            elif avg_accuracy < 0.65:
                print(f"  ○ ACCEPTABLE - Watermark has low detectability")
            else:
                print(f"  ✗ FAIL - Watermark is statistically detectable")
            
            print(f"{'='*70}\n")
        
        return {
            "classifiers": results,
            "average_accuracy": avg_accuracy,
            "is_stealthy": avg_accuracy < 0.55,
            "n_samples": n_samples
        }
    
    def compute_perplexity_difference(self, 
                                     original_texts: List[str],
                                     watermarked_texts: List[str]) -> Dict:
        """
        Compute perplexity difference between original and watermarked texts
        (Requires language model - simplified version here)
        """
        # Simplified: Use word frequency as proxy for perplexity
        def simple_perplexity(text):
            words = re.findall(r'\b\w+\b', text.lower())
            if len(words) == 0:
                return 0
            word_freq = Counter(words)
            probs = np.array(list(word_freq.values())) / len(words)
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            return 2 ** entropy
        
        orig_ppls = [simple_perplexity(t) for t in original_texts]
        wm_ppls = [simple_perplexity(t) for t in watermarked_texts]
        
        avg_orig = np.mean(orig_ppls)
        avg_wm = np.mean(wm_ppls)
        ppl_increase = ((avg_wm - avg_orig) / avg_orig) * 100
        
        return {
            "original_ppl": avg_orig,
            "watermarked_ppl": avg_wm,
            "ppl_increase_percent": ppl_increase,
            "is_acceptable": abs(ppl_increase) < 10  # < 10% change
        }


def test_stealthiness():
    """Test the stealthiness testing module"""
    print("Testing Statistical Stealthiness Module...")
    
    # Generate test data
    original_texts = [
        "The quick brown fox jumps over the lazy dog. This is a test sentence.",
        "Machine learning is a subset of artificial intelligence. It focuses on data.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models have achieved remarkable results in recent years.",
        "The research paper presents a novel approach to the problem."
    ]
    
    # Simulate watermarked texts (slightly modified)
    watermarked_texts = [
        "The quick brown fox jumps over the lazy dog. This is a test sentence.",  # Same
        "Machine learning is a subset of artificial intelligence. It focuses on information.",  # Changed word
        "Natural language processing enables computers to understand human language.",  # Same
        "Deep learning models have achieved remarkable outcomes in recent years.",  # Changed word
        "The research paper presents a novel method to the problem."  # Changed word
    ]
    
    # Test
    tester = StatisticalStealthinessTest()
    results = tester.test_detectability(original_texts, watermarked_texts)
    
    # Test perplexity
    ppl_results = tester.compute_perplexity_difference(original_texts, watermarked_texts)
    print(f"\nPerplexity Analysis:")
    print(f"  Original PPL: {ppl_results['original_ppl']:.2f}")
    print(f"  Watermarked PPL: {ppl_results['watermarked_ppl']:.2f}")
    print(f"  Increase: {ppl_results['ppl_increase_percent']:.2f}%")
    print(f"  Acceptable: {ppl_results['is_acceptable']}")


if __name__ == "__main__":
    test_stealthiness()
