# Knowledge-Guided Dual-Channel Watermarking (KGDW)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **Knowledge-Guided Dual-Channel Watermarking (KGDW)** for LLM-generated text.

KGDW is a **post-processing** watermarking framework: it injects watermark signals into already generated text, **without modifying the LLM decoding process**.

---

## Overview

KGDW uses a **dual-channel architecture**:

- **Primary channel (visible / structured)**: spaCy + Newton + MCP style embedding at syntactically stable positions  
  - Implementation: `methods/enhanced_spacy_newton.py`
- **Secondary channel (invisible / zero-width)**: Unicode / zero-width character (ZWC) embedding with redundancy and optional error correction  
  - Implementation: `methods/localized_unicode.py`

The main evaluation pipeline (embedding → attacks → detection/recovery → reporting) is driven by:

- `improved_main.py` (main experiment runner)
- `experiments/comprehensive_rigorous_suite.py` (ablation + cross-domain/model + combined attacks + statistical analysis + baseline comparison)

Key properties reported in our paper-scale experiments (59,033 samples):

- **Imperceptibility**
  - BLEU ≈ **0.934**, ROUGE-L ≈ **0.947**, semantic similarity ≈ **0.934**
  - ~6.6% degradation vs. pristine text, with stable perplexity
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

## Repository Structure

- `improved_main.py`: end-to-end pipeline (embed/detect/recover, attacks, metrics, per-domain/model summaries, cross-domain/cross-generator stats).
- `methods/`
  - `enhanced_spacy_newton.py`: primary (structured) channel embedding/extraction.
  - `localized_unicode.py`: secondary (ZWC) channel embedding/extraction and dataset profiles.
  - `semantic_anchor.py`, `candidate_selector.py`: supporting utilities.
- `attacks/attack_utils.py`: attack implementations (synonym substitution, deletion/insertion/modification, sentence cutting, physical attacks, burst errors).
- `evaluation/metrics.py`: BLEU/ROUGE, semantic similarity (SBERT), METEOR, perplexity-like, recovery and char-level F1.
- `tools/`
  - `preprocess_data_enhanced.py`: dataset cleaning/preprocessing.
  - `experiment_reporter.py`: formatted reporting (tables/score cards).
- `experiments/`
  - `comprehensive_rigorous_suite.py`: ablations, cross-domain/model, combined attacks, statistical analysis, LOCAT/baseline comparison, human control.
  - `preprocess_human_text.py`: build a human-text control set from input JSONL.
- PowerShell helpers (Windows): `setup_environment.ps1`, `quick_start.ps1`, `preprocess_and_run.ps1`, `run_experiments.ps1`.

---

## Requirements

- Python **≥ 3.8** (recommended: 3.10)
- spaCy English model: `en_core_web_sm`
- Optional GPU: speeds up embedding-based semantic metrics and some heavy attacks
- OS: Windows / Linux / macOS  
  - Windows scripts are provided for quick reproduction (`*.ps1`)

---

## Installation

### Clone

```bash
git clone https://github.com/zyg0326/kgdw-watermark.git
cd kgdw
```

### Create environment

```bash
python -m venv .venv
```

Activate:

- Windows (PowerShell):

```powershell
.\.venv\Scripts\activate
```

- Linux/macOS:

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### (Windows) Optional: one-shot environment setup

```powershell
.\setup_environment.ps1
```

---

## Data Format (Input)

This repository expects input data in **JSONL** files under:

- `data/input/*.jsonl` (raw) or `data/input_cleaned/*.jsonl` (preprocessed)

Each JSON line should include at least one text field. The runner tries the following keys in order:

- `machine_text` (preferred), then `human_text`, then a set of common fallbacks:
  - `text`, `content`, `response`, `output`, `completion`, `generated_text`, `answer`, `prediction`, `generated`

### Filename convention (important)

`improved_main.py` parses **domain** and **model** from the JSONL filename. The recommended format is:

```
<domain>_<model>_*.jsonl
```

Supported domains include: `arxiv`, `reddit`, `wikihow`, `wikipedia` (and several optional extra domains in code).  
Supported model tags include: `chatgpt`, `cohere`, `davinci`, `dolly`, `bloomz`, `flan-t5`.

Example:

```
arxiv_chatgpt_000.jsonl
wikihow_dolly_001.jsonl
```

If the domain/model cannot be parsed from filename, the file will be skipped.

---

## Experimental Evaluation (Reproduction Guide)

This section mirrors the paper’s evaluation goals:

1. **Text quality preservation** (imperceptibility)
2. **Robustness under diverse attacks** (including structural and semantic attacks)
3. **Model-agnostic generalization** across different LLMs and domains

### Datasets and Models (Paper Setting)

We evaluate on four domains from the **M4 benchmark** (total **59,033** samples):

- **arXiv**: 15,000
- **Reddit**: 18,000
- **WikiHow**: 15,000
- **Wikipedia**: 11,033

We evaluate five generators:

- **GPT-3.5-Turbo**: 11,995
- **Cohere Command**: 11,336
- **GPT-3 Davinci**: 12,000
- **Dolly-v2-12B**: 11,702
- **FLAN-T5-XXL**: 6,000

### Metrics

The code computes (see `evaluation/metrics.py`):

- **BLEU**, **ROUGE-L**, **METEOR**
- **Semantic similarity** via Sentence-BERT embeddings
- **Perplexity-like** score (GPT-2 style)
- **Detection metrics**: accuracy / precision / recall / F1 (via confusion matrix)
- **Recovery**: character-level recovery/F1 of extracted watermark payload

### Attacks

Attacks are applied in `improved_main.py` and implemented in `attacks/attack_utils.py`, including:

- Deletion / insertion / modification (10–50% intensity)
- Burst errors (character block deletions)
- Sentence cutting / structural corruption
- Synonym substitution (10–50%)
- Optional physical attacks: print-scan, screenshot (can be toggled)

The overall attack intensity can be controlled via `WM_ATTACK_PROB` (0.0–1.0).

---

## Quick Start (Sanity Check)

Runs a tiny experiment (3 samples, skips semantic similarity and all attacks) to verify your environment:

```powershell
.\quick_start.ps1
```

This calls `python improved_main.py` with:

- `WM_MAX_LINES=3`
- `WM_SKIP_SEMANTIC=true`
- `WM_SKIP_ATTACKS=true`

---

## Full Pipeline (Preprocess + Main Experiment)

### Option A: One command (recommended on Windows)

```powershell
.\preprocess_and_run.ps1
```

What it does:

1. Preprocess raw data: `tools/preprocess_data_enhanced.py`  
   - input: `data/input`  
   - output: `data/input_cleaned`
2. Clear `data/cache/*`
3. Run the main experiment: `python improved_main.py`

### Option B: Manual steps (cross-platform)

1) Preprocess:

```bash
python tools/preprocess_data_enhanced.py --input "data/input" --output "data/input_cleaned" --max_samples 0
```

> Notes:
> - `--max_samples 0` means “no limit” if supported by your script; if your local run expects a positive integer, set a large number instead.
> - You can also skip preprocessing and point the runner to `data/input` (see `INPUT_DIR` in `improved_main.py`).

2) Run main experiment:

```bash
python improved_main.py
```

---

## Full Benchmark Script (Windows)

`run_experiments.ps1` runs:

- **Experiment A**: quick verification (small sample, low-cost)
- **Experiment B**: full benchmark run (higher intensity, report/log export)

```powershell
.\run_experiments.ps1
```

Logs:

- `experiment_log_full.txt` (full run console log)

---

## Comprehensive Rigorous Suite (Ablations + Baselines)

To reproduce detailed ablations, combined attacks, and LOCAT/baseline comparisons:

```bash
python experiments/comprehensive_rigorous_suite.py
```

Outputs:

- `data/output/comprehensive_<timestamp>/comprehensive_results.json`

This suite includes:

- Multi-condition ablation on:
  - anchor count, ZWC count, detection thresholds
- Cross-domain and cross-model breakdowns
- Combined attack scenarios
- Statistical analysis (multiple runs, mean/std, 95% CI)
- Comparison against LOCAT (Robust/Gentle), DeepTextMark, and Yang et al. (using paper-reported baselines embedded in the script)
- Human text control experiment (FPR test)

---

## Human Text Control Set (Optional)

Build a human-text control set (for FPR / specificity testing):

```bash
python experiments/preprocess_human_text.py
```

Output directory:

- `data/input_human_text/`

You can then run `improved_main.py` on this directory by adjusting `INPUT_DIR` in `improved_main.py` (or by creating a small wrapper).

---

## Environment Variables (Common Knobs)

Most experiment knobs are exposed via environment variables (read by `improved_main.py` and scripts):

- `PREPROC_GROUP`  
  - Controls preprocessing / watermark group behavior. Example: `base`, `base32_crc_spacy_newton`.
- `WM_MAX_LINES`  
  - Per-file sample cap (0 = all).
- `WM_ATTACK_PROB`  
  - Global attack intensity in \[0, 1\]. Higher means stronger deletion/insertion/modification/synonym substitution ratios.
- `WM_SKIP_SEMANTIC`  
  - `true/false`. Skips SBERT semantic similarity (faster).
- `WM_SKIP_ATTACKS`  
  - `true/false`. Skips all attacks (faster).
- `WM_USE_PHYSICAL_ATTACK`  
  - `true/false`. Enables print-scan / screenshot attacks.
- `WM_SKIP_SLOW_ATTACKS`  
  - `true/false`. Skips slow attacks like synonym substitution and physical attacks.

Example (PowerShell):

```powershell
$env:PREPROC_GROUP="base32_crc_spacy_newton"
$env:WM_ATTACK_PROB="0.8"
$env:WM_MAX_LINES="50"
python improved_main.py
```

---

## Outputs (What to Look At)

After running `improved_main.py`, the key outputs are:

- `data/output/improved_watermark_results.json`  
  - Per-sample records: watermarked text, clean extraction, attack extraction, and quality metrics.
- `data/output/improved_watermark_summary.json`  
  - Aggregate success rates by method (`enhanced`, `unicode`, `combined`), attack survival, and detection metrics.
- `data/output/cross_metrics.json`  
  - Cross-domain transfer rates and cross-generator decay rates computed from per-domain/model success.

Some scripts may also export CSV summaries to:

- `data/report_csv/` (if export scripts are available/enabled)

---

## Results Discussion (High-Level)

Our experiments support three key findings:

- **Imperceptibility**: Watermarked texts preserve semantic meaning and surface form, achieving high BLEU/ROUGE and high semantic similarity with stable perplexity.
- **Robustness**: Detection and recovery remain strong under multiple attacks (deletion/insertion/modification/burst errors and physical-style transformations).
- **Semantic attack resistance**: Synonym substitution (a core challenge for lexical watermarking) has minimal effect on the secondary ZWC channel, leading to near-perfect detection under heavy substitution.

Domain trends observed in our analysis:

- **WikiHow** tends to yield the best fidelity due to stable, template-like structure.
- **Wikipedia** often shows strong robustness due to more formal syntax.
- **Reddit** is more challenging (higher entropy), but remains robust under attacks.
- **arXiv** sits in the middle, with structure and terminology affecting anchor stability.

Across models, metrics remain stable, supporting **model-agnostic generalization**.

---

## Troubleshooting

- **spaCy model not found**

```bash
python -m spacy download en_core_web_sm
```

- **Windows console encoding issues**
  - Use the provided PowerShell scripts (`*.ps1`) which set UTF-8 (`chcp 65001`).

- **Slow runtime**
  - Disable heavy parts:
    - `WM_SKIP_SEMANTIC=true`
    - `WM_SKIP_SLOW_ATTACKS=true`
    - Reduce `WM_MAX_LINES`

---

## License

This project is released under the **MIT License**. See `LICENSE`.
