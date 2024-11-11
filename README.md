# Using Language Models to Disambiguate Lexical Choices in Translation

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](http://arxiv.org/abs/2411.05781)
[![Conference](https://img.shields.io/badge/Conference-EMNLP%202024-blue)](https://2024.emnlp.org/)

## Authors
- [Josh Barua](https://joshbarua.github.io/)
- [Sanjay Subramanian](https://people.eecs.berkeley.edu/~sanjayss/)
- [Kayo Yin](https://kayoyin.github.io/)
- [Alane Suhr](https://www.alanesuhr.com/)

*UC Berkeley*

## News 🎉
- **September 20, 2024**: Paper accepted at EMNLP 2024 main proceedings!

## Project Overview

This project investigates whether modern large language models (LLMs) understand concept variation across languages. We focus on the task of lexical selection—choosing the most suitable lexeme in the target language to match a single lexeme in the source language, based on the context of the source sentence. 

To support this study, we introduce DTAiLS, a dataset covering nine languages with expert-verified sentence pairs that demonstrate concept variation in translation. Additionally, we explore the impact of lexical rules (natural language descriptions of concept variation) on model performance during lexical selection. 

Our key takeaways are:
- All LLMs can reason over source-side context to resolve ambiguity given high-quality rules, but only the strongest closed-weight LLMs can generate such rules
- Lexical rules can help bridge gaps in knowledge of concept variation in low-resource languages across models 

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Generating Lexical Rules](#generating-lexical-rules)
- [Model Evaluation](#model-evaluation)
- [Computing Results](#computing-results)
- [Citation](#citation)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Berkeley-NLP/Lex-Rules.git
cd Lex-Rules
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate lex-rules
```

3. Visit the [PyTorch website](https://pytorch.org/get-started/locally/) and install torch locally. I used the following command:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

4. To use GPT-4-Turbo, set your API key:
```bash
export OPENAI_API_KEY=<your-api-key>
```

## Dataset

The DTAiLS dataset is located in the `./data/` directory, organized by language pairs (e.g., `en-af` for English-Afrikaans).

## Generating Lexical Rules

Generate lexical rules using language models with the following script:

```bash
python generate_lexical_rules.py \
    --model="gpt-4-turbo" \
    --source_language="English" \
    --target_language="Afrikaans" \
    --source_concepts="data/en-af/af.words" \
    --out_file="lexical_rules/gpt-4-turbo-afrikaans.csv" \
    --use_incontext_sentences \
    --num_sentences=25 \
    --train_dataset="data/en-af/train_dataset.csv" \
    --max_retries=3
```

### Parameters

| Parameter | Description | Options |
|-----------|-------------|----------|
| `--model` | Model to use | `gpt-4-turbo`, `llama-3-8b`, `gemma-7b` |
| `--source_concepts` | Path to source concepts file | Path String |
| `--out_file` | Output file path | Path String |
| `--use_incontext_sentences` | Include example sentences in prompt | Flag |
| `--num_sentences` | Number of example sentences per variation | Integer |
| `--train_dataset` | Path to training dataset | Path String |

## Model Evaluation

### Language Model Evaluation

Evaluate language models on lexical selection:

```bash
python evaluate.py \
    --model="gpt-4-turbo" \
    --target_language="Afrikaans" \
    --test_dataset="data/en-af/expert_dataset.csv" \
    --eval_mode="lexical_rules" \
    --lexical_rules="lexical_rules/gpt-4-turbo-afrikaans.csv" \
    --out_file="results/afrikaans_gpt-4-turbo_predictions.csv" \
    --results_file="data/en-af/results.csv" \
    --results_file_column="GPT4" \
    --shuffle_order \
    --max_retries=2
```

#### Parameters

| Parameter | Description | Options |
|-----------|-------------|----------------|
| `--model` | Model to use for evaluation | `gpt-4-turbo`, `llama-3-8b`, `gemma-7b` |
| `--target_language` | Target language for translation | String (e.g., "Afrikaans") |
| `--test_dataset` | Path to test dataset | Path String |
| `--eval_mode` | Evaluation mode | `lexical_rules`, `baseline` |
| `--lexical_rules` | Path to lexical rules file | Path String |
| `--out_file` | Output file for predictions | Path String |
| `--results_file` | File for accumulating results | Path String |
| `--results_file_column` | Column name in results file | `GPT4`, `GPT4-NoRules`, `Llama-3-8b-GPT4Rules`, `Llama-3-8b`, `Llama-3-8b-NoRules`, `Gemma-7b-GPT4Rules`, `Gemma-7b`, `Gemma-7b-NoRules` |
| `--shuffle_order` | Randomize order of lexical variations in prompt | Flag |
| `--max_retries` | Maximum attempts to generate valid response | Integer |

### Neural MT Evaluation

Evaluate neural machine translation models:

```bash
python mt_evaluate.py \
    --model="madlad400-10b" \
    --test_dataset="data/en-af/expert_dataset.csv" \
    --target_lang_id="af" \
    --results_file="data/en-af/results.csv" \
    --column_name="MADLAD400-10b"
```

#### Parameters

| Parameter | Description | Options |
|-----------|-------------|----------------|
| `--model` | MT model to evaluate | `madlad400-10b`, `nllb-200-3.3b` |
| `--test_dataset` | Path to test dataset | Path String |
| `--target_lang_id` | Target language code | Two-letter code (see supported languages table) |
| `--results_file` | File for accumulating results | Path String |
| `--column_name` | Column name in results file | `MADLAD400-10b`, `NLLB-200-3.3b` |

### Supported Models and Languages

**Models:**
- Language Models: `gpt-4-turbo`, `llama-3-8b`, `gemma-7b`
- MT Systems: `madlad400-10b`, `nllb-200-3.3b`

**Target Languages:**
| Code | Language |
|------|-----------|
| af | Afrikaans |
| ta | Tamil |
| te | Telugu |
| gl | Galician |
| hi | Hindi |
| hy | Armenian |
| ja | Japanese |
| fa | Farsi |
| lv | Latvian |

## Computing Results

Calculate micro-averages across all concepts:

```bash
LANGS=("afrikaans" "tamil" "telugu" "galician" "hindi" "armenian" "japanese" "farsi" "latvian")
LANG_IDS=("en-af" "en-ta" "en-te" "en-gl" "en-hi" "en-hy" "en-ja" "en-fa" "en-lv")
for (( i=0; i<9; i++ )); do
    python computeOverallAccuracy.py \
        --results_file="data/${LANG_IDS[$i]}/results.csv" \
        --output_file="accuracy.csv" \
        --language=${LANGS[$i]} 
done
```

## Citation

Please consider citing our paper if you find our work or dataset helpful for your research:

```bibtex
@inproceedings{barua24emnlp,
    title = {Using Language Models to Disambiguate Lexical Choices in Translation},
    author = {Barua, Josh and Subramanian, Sanjay and Yin, Kayo and Suhr, Alane},
    booktitle = {EMNLP},
    month = {November},
    year = {2024}
}
```