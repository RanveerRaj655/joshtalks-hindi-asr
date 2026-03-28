# Josh Talks - Hindi ASR Pipeline

This repository contains an end-to-end Hindi Automatic Speech Recognition (ASR) pipeline built for the AI Researcher Intern Assignment at Josh Talks. The project focuses on fine-tuning, robust text post-processing, scalable Devanagari spell classification, and advanced Word Error Rate (WER) evaluations.

*(Note: Raw 10-hour audio datasets, large model checkpoints, and cache directories have been excluded from this repository due to file-size constraints. This repository contains the core codebase and resulting metrics.)*

---

## 📌 Core Features

1. **ASR Fine-Tuning Pipeline (`prepare_dataset.py`, `train_whisper.py`)**  
   Downloads, resamples, and chunks 10 hours of Hindi conversational audio, and fine-tunes `openai/whisper-small` using HuggingFace `Seq2SeqTrainer` and LoRA/gradient-checkpoint optimizations for efficiency.

2. **Error Taxonomy & Analysis (`error_analysis.py`)**  
   Performs a systematic, 25-sample cross-sectional analysis over three WER severity brackets to categorize ASR failure modes (Matras, Conjuncts, English Loanwords, etc.).

3. **Transcript Post-Processing (`post_processing.py`)**  
   - **Number Normalization:** Translates Hindi written numbers (e.g., "एक लाख चौवन") to digits (`100054`), while intelligently keeping idiomatic phrases untouched ("दो-चार बातें").
   - **Code-Switching Tagger:** Accurately isolates English loanwords using Roman script analysis and phonetic translation, bounded by a 55-word whitelist to prevent tagging fully-adopted loanwords (e.g., डॉक्टर).

4. **Devanagari Spell Classifier (`spell_classifier.py`)**  
   A pure heuristics-based classifier that evaluated 177,421 unique Hindi tokens. Utilizes 4 distinct Hindi lexicons (including Hunspell) combined with custom multi-stage phonotactic constraint validations.

5. **Lattice-Based WER System (`lattice_wer.py`)**  
   Mitigates human-annotation flaws by utilizing Levenshtein dynamic programming. Aligns 6 separate model hypotheses onto a single human spine and dynamically substitutes words where democratic model consensus overrules human transcript error.

---

## ⚙️ Installation

To set up the environment, clone the repository and install the required dependencies:

```bash
git clone https://github.com/RanveerRaj655/joshtalks-hindi-asr.git
cd joshtalks-hindi-asr

pip install -r requirements.txt
# Alternatively: pip install transformers datasets==2.14.7 torch jiwer evaluate librosa soundfile pandas numpy
```

---

## 🚀 How to Run the Pipeline

The project is structured logically across the 4 assignment questions. Run the following code sequentially from the root directory to replicate the pipeline outputs:

**1. Data Prep, Fine-Tuning & Standard Evaluation (Q1)**
```bash
python prepare_dataset.py   # Synthesizes HF Dataset from raw GCS links
python train_whisper.py     # Fine-tunes openai/whisper-small (Requires GPU)
python evaluate_whisper.py  # Computes WER on test splits against reference
python error_analysis.py    # Stratifies and outputs root error typography
```

**2. Transcripts Post-Processing Pipeline (Q2)**
```bash
python post_processing.py
```
*(Runs rule-based conversions and English [EN] tagging constraints on sample transcripts)*

**3. Large-Scale Spell Classification (Q3)**
```bash
python spell_classifier.py
```
*(Processes 177K inputs and categorizes spelling accuracy using multi-stage logic into output CSVs)*

**4. Lattice-Based WER Framework (Q4)**
```bash
python lattice_wer.py
```
*(Parses `Question 4 - Task.csv`, establishes lattice bins, applies consensus logic, and prints the overall Delta performance of the models)*

---

