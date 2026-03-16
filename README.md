# Knowledge-Guided Dual-Channel Watermarking (KGDW)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the **Knowledge-Guided Dual-Channel Watermarking (KGDW)** framework for LLM-generated text, as described in our paper (submitted to *Knowledge-Based Systems*).

KGDW is a **post-processing** watermarking framework that injects signals into already generated text, without modifying the LLM’s decoding process.

---

## 📋 Overview

KGDW employs a **dual-channel architecture**:

- **Primary channel (visible / structured)**  
  A spaCy + Newton + MCP based channel (`methods/enhanced_spacy_newton.py`) that embeds watermark symbols at syntactically stable positions.

- **Secondary channel (invisible / zero-width)**  
  A Unicode / zero-width character (ZWC) channel (`methods/localized_unicode.py`) that provides high-capacity, error-correctable embedding in the encoding layer.

Key properties (from our experiments on 59,033 samples):

- **Imperceptibility**  
  - BLEU ≈ **0.934**, ROUGE-L ≈ **0.947**, semantic similarity ≈ **0.934**  
  - Only ~6.6% degradation vs pristine text, with stable perplexity

- **Robustness**  
  - Clean detection accuracy ≈ **0.990**, clean recovery ≈ **0.991**  
  - Under attacks: accuracy ≈ **0.899**, recovery ≈ **0.907**  
  - F1 = **1.00** for several attack types; strong resilience under word deletion

- **Semantic attack resistance**  
  - Under heavy synonym substitution (50%), F1 ≈ **0.998**  
  - **+18.8 percentage points** improvement over LOCAT-Robust

- **Cross-domain & cross-generator generalization**  
  - Cross-domain transfer rate: **1.0000** (44 domain pairs)  
  - Cross-generator transfer rate: **1.0000** (58 model pairs)

---

## 🚀 Quick Start

### Prerequisites

- Python ≥ 3.8 (recommended 3.10)
- `pip` to install dependencies
- spaCy English model `en_core_web_sm`
- CUDA-capable GPU (optional, for SBERT/NLI-based attacks and faster evaluation)
- OS: Windows / Linux / macOS  
  (Windows has additional PowerShell helpers in this repo)

### Installation

# Clone the repository
git clone https://github.com/zyg0326/kgdw-watermark.git
cd kgdw

# (Optional but recommended) Create a virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm
