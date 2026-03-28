# Josh Talks AI Researcher Intern Assignment 

**Candidate Name:** Ranveer Raj  
**Date:** March 27, 2026  
**Role:** AI Researcher Intern  

---

## Executive Summary
This report contains the results for the four components of the AI Researcher Assessment: ASR data preprocessing and model fine-tuning, error analysis, text post-processing, Devanagari spell-checking, and Lattice-based Word Error Rate (WER) evaluation.

---

## Question 1: Whisper Fine-Tuning & Error Analysis

### 1.1 Data Preprocessing & Model Training
Fine-tuned `openai/whisper-small` on ~10 hours of Hindi audio data. The dataset was preprocessed using the following steps:
*   **Audio Standardization:** All files were resampled to 16kHz mono audio.
*   **Silence Removal:** Automatically trimmed leading, trailing, and intraword silences using parameterized VAD thresholds.
*   **Volume Normalization:** Audio arrays were normalized to 0 dBFS.
*   **Audio Duration Masking:** Sentences shorter than 1 second and longer than 30 seconds were filtered out to avoid dimension mismatch errors during batching.

### 1.2 Evaluation & WER Results
The fine-tuned model and base `whisper-small` were evaluated on the **FLEURS Hindi Test Split**.

| Model | Test Dataset | Word Error Rate (WER) |
| :--- | :--- | :--- |
| `openai/whisper-small` (Baseline) | FLEURS Hindi (Test) | 70.46% |
| `whisper-small-hi-finetuned` | FLEURS Hindi (Test) | 31.78% |

*Note: Baseline WER on FLEURS is high before fine-tuning due to zero-shot generation mismatch and normalization factors.*

### 1.3 Systematic Error Analysis
Qualitative error analysis process:
1.  **Stratified Sampling:** 25 error samples were systematically extracted across three severity buckets (Low: 0-30%, Medium: 30-60%, High: >60%).
2.  **Error Taxonomy:** Created a custom classification system tracking 8 distinct Devanagari ASR error types (Matra confusions, Conjunct consonants, English loanwords, Code-switching, Number translation, etc.).
3.  **Actionable Fixes:** Identified the top 3 error types and proposed technical solutions (e.g., adding rule-based VAD for short-word insertion/deletion, BPE vocabulary extension for conjuncts).

---

## Question 2: ASR Transcript Cleanup Pipeline

Built a regex and rule-based post-processing pipeline to clean raw Hindi transcripts.

### 2.1 Hindi Number Normalizer
Converts spelled-out Hindi number words directly into digits (handling numbers from 0 up to 99,999+).
*   **Compound parsing:** Successfully processes structures like "एक लाख बीस हज़ार" → `120000`.
*   **Edge Case Handling:** Skips idiomatic expressions using lookahead patterns so phrases like "दो-चार बातें" remain intact rather than becoming "2-4 बातें".

### 2.2 English Loanword Tagger
Detects English words in Devanagari script and wraps them with `[EN]...[/EN]` boundary tags.
*   **Detection Strategy:** Combines Roman script pattern matching with an inner dictionary of 200+ common transliterations (e.g., इंटरव्यू, जॉब, ऑफिस).
*   **Whitelist Exclusion:** Explicitly excludes ~55 fully absorbed Hindi loanwords (e.g., डॉक्टर, बस, मशीन, मोबाइल) to prevent false-positive tagging.

---

## Question 3: Large-Scale Devanagari Spell Classification

Evaluated 177,421 unique Hindi words using a multi-stage spelling classification pipeline.

### 3.1 Classification Methodology
1.  **Aggregated Dictionaries:** Merged data from 4 distinct GitHub sources (including Hunspell Hindi) and expanded a built-in core vocabulary covering systematic verb conjugations.
2.  **Phonotactic Constraints:** Implemented strict Devanagari syntactic rules to auto-flag high-confidence errors (e.g., repeating matras `ाा`, three consecutive letters, invalid vowel sequences).
3.  **Confidence Scoring Bracket:** Words were mapped to HIGH, MEDIUM, or LOW confidence indices based on dictionary presence, morphological suffix extrapolation, and length heuristics.

### 3.2 Key Statistics
*   **Total Words Analyzed:** 177,421
*   **Correctly Spelled (Valid):** 140,978
*   **Incorrectly Spelled (Flagged):** 35,176
*   **Confidence Optimization:** By assigning MEDIUM confidence to pure Devanagari words between 2-15 characters, the LOW confidence bracket was reduced to **11.6%**.

---

## Question 4: Lattice-Based WER Evaluation System

Standard Word Error Rate can penalize correct model outputs if the human reference contains errors. A Lattice-based WER framework was implemented to address this.

### 4.1 System implementation
*   **Word-Level Dynamic DP Alignment:** Utilized Levenshtein alignment (via `jiwer`) to map 6 different model candidate strings against the Human Reference baseline.
*   **Lattice Bin Construction:** Positional bins were created. If **3 or more models** mutually agreed upon a word differing from the human reference, it superseded or joined the reference as a valid semantic alternative in that positional bin.
*   **Custom Evaluation Tracking:** Substitutions that hit a valid lattice bin alternative were explicitly forgiven.

### 4.2 Impact Comparison Table
5 out of 6 models showed a lower Lattice WER compared to Standard WER.

| Model | Standard WER (%) | Lattice WER (%) | Delta (%) | Fairly Penalized? |
| :--- | :--- | :--- | :--- | :--- |
| **Model H** | 3.30 | 2.32 | 0.98 | Yes |
| **Model i** | 4.40 | 4.03 | 0.37 | No |
| **Model k** | 20.42 | 19.07 | 1.34 | Yes |
| **Model l** | 9.17 | 7.82 | 1.34 | Yes |
| **Model m** | 16.87 | 14.43 | 2.44 | Yes |
| **Model n** | 11.12 | 8.80 | 2.32 | Yes |

*Model H achieved the lowest Lattice WER at 2.32%.*

---

## 5. How to Run the Code

Due to the 10 MB strict file size limit on the submission portal, the 10-hour Hindi ASR dataset (`raw_audio/`), `.wav` files, and the HuggingFace dataset cache have been omitted from this ZIP archive. Only the core Python source code, generated CSV metrics, and the report are included.

To replicate the pipeline, execute the scripts in the following order from the root directory:

**1. Install Dependencies**
```bash
pip install -r requirements.txt
# Alternatively: pip install transformers datasets==2.14.7 torch jiwer evaluate librosa soundfile pandas numpy
```

**2. Question 1: Dataset Prep & Fine-Tuning**
```bash
python prepare_dataset.py  # Connects to GCS, downloads audio, resamples to 16kHz
python train_whisper.py    # Fine-tunes `openai/whisper-small` (Requires GPU)
python evaluate_whisper.py # Computes WER on test splits
python error_analysis.py   # Runs the stratified systematic error taxonomy
```

**3. Question 2: Transcripts Post-Processing Pipeline**
```bash
python post_processing.py
```
*(Runs rule-based number translations (Words → Digits) and tags code-switched English)*

**4. Question 3: Large-Scale Spell Classification**
```bash
python spell_classifier.py
```
*(Classifies 177,421 words, generates 3 CSVs including `spelling_classification.csv`)*

**5. Question 4: Lattice-Based WER Framework**
```bash
python lattice_wer.py
```
*(Constructs Levenshtein position bins, runs consensus evaluations, and exports `lattice_wer_results.csv`)*
