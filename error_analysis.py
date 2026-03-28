import argparse
import re
import pandas as pd
import jiwer
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).parent
DEFAULT_CSV = str(BASE_DIR / "predictions_whisper-small_baseline.csv")


#  Hindi-specific character sets 

MATRAS = set("ा ि ी ु ू े ै ो ौ ं ः ँ ृ".split())
CONSONANTS = set("क ख ग घ ङ च छ ज झ ञ ट ठ ड ढ ण त थ द ध न प फ ब भ म "
                  "य र ल व श ष स ह".split())
HALANT = "्"
DIGITS_HI = set("०१२३४५६७८९")
DIGITS_EN = set("0123456789")

# Common English loanwords in Devanagari
ENGLISH_MARKERS = re.compile(
    r"[a-zA-Z]{2,}|"  # Latin script words
    r"(?:कम्प्यूटर|मोबाइल|फोन|इंटरनेट|ऑनलाइन|ऑफलाइन|"
    r"डाउनलोड|अपलोड|सॉफ्टवेयर|हार्डवेयर|डिजिटल|"
    r"पॉलिसी|रिपोर्ट|प्रोजेक्ट|मैनेजर|मीटिंग|"
    r"स्कूल|कॉलेज|यूनिवर्सिटी|हॉस्पिटल|"
    r"बस|ट्रेन|एयरपोर्ट|टिकट|होटल)"
)


#  TASK 1 — Systematic Sampling

def task1_systematic_sampling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by WER, bucket into severity levels, sample ~8-9 from each → 25 total.
    """
    print("\n" + "=" * 80)
    print("  TASK 1: SYSTEMATIC SAMPLING BY SEVERITY")
    
    # Assign severity buckets
    df = df.copy()
    df["severity"] = pd.cut(
        df["wer_percent"],
        bins=[-0.1, 30, 60, float("inf")],
        labels=["Low (0-30%)", "Medium (30-60%)", "High (>60%)"],
    )

    # Sort by WER within each bucket
    df = df.sort_values("wer_percent")

    # Print bucket distribution
    print("\n  Bucket distribution:")
    for sev, count in df["severity"].value_counts().sort_index().items():
        print(f"    {sev}: {count} samples")

    # Sample ~8-9 per bucket, total = 25
    samples_per_bucket = {
        "Low (0-30%)": 9,
        "Medium (30-60%)": 8,
        "High (>60%)": 8,
    }

    sampled_frames = []
    for severity, n in samples_per_bucket.items():
        bucket = df[df["severity"] == severity]
        if len(bucket) == 0:
            print(f"\n  ⚠ No samples in bucket: {severity}")
            continue
        n_actual = min(n, len(bucket))
        # Evenly space the samples across the bucket
        indices = bucket.index.tolist()
        step = max(1, len(indices) // n_actual)
        selected = [indices[i] for i in range(0, len(indices), step)][:n_actual]
        sampled_frames.append(df.loc[selected])

    sampled = pd.concat(sampled_frames).reset_index(drop=True)
    sampled["id"] = range(1, len(sampled) + 1)

    # Print formatted table
    print(f"\n  Selected {len(sampled)} samples:\n")
    print(f"  {'ID':>3} │ {'WER':>6} │ {'Severity':<15} │ {'Reference':<35} │ {'Hypothesis':<35}")
    print(f"  {'─'*3} │ {'─'*6} │ {'─'*15} │ {'─'*35} │ {'─'*35}")

    for _, row in sampled.iterrows():
        ref = row["reference"][:33] + "…" if len(str(row["reference"])) > 35 else str(row["reference"])
        hyp = row["prediction"][:33] + "…" if len(str(row["prediction"])) > 35 else str(row["prediction"])
        print(f"  {row['id']:>3} │ {row['wer_percent']:>5.1f}% │ {row['severity']:<15} │ {ref:<35} │ {hyp:<35}")

    # Save sampled data
    out_path = BASE_DIR / "error_samples_25.csv"
    sampled.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  → Saved to: {out_path}")

    return sampled


#  TASK 2 — Error Taxonomy

def _classify_error(ref: str, hyp: str) -> list[str]:
    """
    Analyze a single (reference, hypothesis) pair and return a list of
    error categories that apply.
    """
    categories = []
    ref_words = ref.split()
    hyp_words = hyp.split()

    # 1. Matra / vowel marker errors
    #    Strip consonants → compare remaining matras
    ref_matras = [c for c in ref if c in MATRAS]
    hyp_matras = [c for c in hyp if c in MATRAS]
    if ref_matras != hyp_matras:
        # Check if the consonant skeleton is similar
        ref_cons = re.sub(r"[ािीुूेैोौंःँृ]", "", ref)
        hyp_cons = re.sub(r"[ािीुूेैोौंःँृ]", "", hyp)
        if ref_cons == hyp_cons and ref != hyp:
            categories.append("Matra/Vowel Marker Error")
        elif abs(len(ref_matras) - len(hyp_matras)) > 2:
            categories.append("Matra/Vowel Marker Error")

    # 2. Conjunct consonant errors (halant based)
    ref_conjuncts = ref.count(HALANT)
    hyp_conjuncts = hyp.count(HALANT)
    if abs(ref_conjuncts - hyp_conjuncts) >= 1:
        # Additionally check if nearby characters differ
        if ref_conjuncts != hyp_conjuncts:
            categories.append("Conjunct Consonant Error")

    # 3. Number / digit handling
    ref_has_digits = bool(DIGITS_HI.intersection(ref) or DIGITS_EN.intersection(ref))
    hyp_has_digits = bool(DIGITS_HI.intersection(hyp) or DIGITS_EN.intersection(hyp))
    # Check for number words vs digits
    number_words = {"एक", "दो", "तीन", "चार", "पांच", "छह", "सात", "आठ",
                    "नौ", "दस", "सौ", "हजार", "लाख", "करोड़"}
    ref_has_numwords = bool(number_words.intersection(ref_words))
    hyp_has_numwords = bool(number_words.intersection(hyp_words))
    if (ref_has_digits != hyp_has_digits) or (ref_has_numwords != hyp_has_numwords):
        categories.append("Number/Digit Handling")

    # 4. English loanwords
    ref_eng = set(ENGLISH_MARKERS.findall(ref))
    hyp_eng = set(ENGLISH_MARKERS.findall(hyp))
    if ref_eng != hyp_eng or (ref_eng and hyp_eng and ref_eng.symmetric_difference(hyp_eng)):
        categories.append("English Loanword Error")
    # Also check for Latin script in hypothesis but not reference (or vice versa)
    ref_latin = set(re.findall(r"[a-zA-Z]+", ref))
    hyp_latin = set(re.findall(r"[a-zA-Z]+", hyp))
    if ref_latin != hyp_latin and "English Loanword Error" not in categories:
        categories.append("English Loanword Error")

    # 5. Insertion / deletion of short words (≤2 chars)
    ref_short = {w for w in ref_words if len(w) <= 2}
    hyp_short = {w for w in hyp_words if len(w) <= 2}
    short_diff = ref_short.symmetric_difference(hyp_short)
    if len(short_diff) >= 1:
        categories.append("Short Word Insertion/Deletion")

    # 6. Word-level substitution (words in ref not in hyp or vice versa)
    ref_set = set(ref_words)
    hyp_set = set(hyp_words)
    subs = ref_set.symmetric_difference(hyp_set)
    long_subs = {w for w in subs if len(w) > 3}
    if long_subs and not categories:
        categories.append("Word Substitution")

    # 7. Repetition errors (same word repeated in hypothesis)
    for i in range(len(hyp_words) - 1):
        if hyp_words[i] == hyp_words[i + 1] and hyp_words[i] not in ref:
            categories.append("Repetition/Hallucination")
            break

    # 8. Proper noun confusion
    # Simple heuristic: words starting with upper case or uncommon Devanagari clusters
    if len(long_subs) >= 1:
        for word in long_subs:
            if word[0].isupper() or (len(word) > 4 and word not in ref_set and word not in hyp_set):
                categories.append("Proper Noun Confusion")
                break

    if not categories:
        categories.append("Other/Phonetic Similarity")

    return list(dict.fromkeys(categories))  # deduplicate preserving order


def task2_error_taxonomy(sampled: pd.DataFrame):
    """Classify each sample into error categories and print taxonomy."""
    print("\n" + "=" * 80)
    print("  TASK 2: ERROR TAXONOMY")
    
    # Classify all samples
    all_categories = Counter()
    sample_categories = []
    category_examples = defaultdict(list)

    for _, row in sampled.iterrows():
        ref = str(row["reference"])
        hyp = str(row["prediction"])
        cats = _classify_error(ref, hyp)
        sample_categories.append(cats)

        for cat in cats:
            all_categories[cat] += 1
            if len(category_examples[cat]) < 5:
                category_examples[cat].append({
                    "reference": ref,
                    "hypothesis": hyp,
                    "wer": row["wer_percent"],
                })

    # Print distribution
    print("\n  Error Category Distribution (across 25 samples):\n")
    print(f"  {'Category':<35} {'Count':>5} {'% of Samples':>12}")
    print(f"  {'─'*35} {'─'*5} {'─'*12}")
    for cat, count in all_categories.most_common():
        pct = count / len(sampled) * 100
        bar = "█" * int(pct / 5)
        print(f"  {cat:<35} {count:>5} {pct:>10.0f}%  {bar}")

    # Print detailed examples for each category
    for cat, count in all_categories.most_common():
        examples = category_examples[cat]
        print(f"\n  ┌─ {cat} ({count} occurrences)")
        print(f"  │")
        for i, ex in enumerate(examples[:5], 1):
            ref_short = ex["reference"][:60] + "…" if len(ex["reference"]) > 60 else ex["reference"]
            hyp_short = ex["hypothesis"][:60] + "…" if len(ex["hypothesis"]) > 60 else ex["hypothesis"]
            print(f"  │  Example {i} (WER: {ex['wer']:.1f}%)")
            print(f"  │    REF : {ref_short}")
            print(f"  │    HYP : {hyp_short}")

            # Show specific differences
            ref_words = ex["reference"].split()
            hyp_words = ex["hypothesis"].split()
            added = set(hyp_words) - set(ref_words)
            deleted = set(ref_words) - set(hyp_words)
            if deleted:
                print(f"  │    DEL : {' | '.join(list(deleted)[:5])}")
            if added:
                print(f"  │    ADD : {' | '.join(list(added)[:5])}")
            print(f"  │")
        print(f"  └─")

    # Save taxonomy
    taxonomy_rows = []
    for i, (_, row) in enumerate(sampled.iterrows()):
        taxonomy_rows.append({
            "id": i + 1,
            "reference": row["reference"],
            "hypothesis": row["prediction"],
            "wer_percent": row["wer_percent"],
            "severity": row["severity"],
            "error_categories": " | ".join(sample_categories[i]),
        })
    taxonomy_df = pd.DataFrame(taxonomy_rows)
    out_path = BASE_DIR / "error_taxonomy.csv"
    taxonomy_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  → Taxonomy saved to: {out_path}")

    return all_categories, category_examples


#  TASK 3 — Actionable Fixes

FIXES = {
    "Matra/Vowel Marker Error": {
        "fix": "Phoneme-Aware Post-Processing with Soundex-like Mapping",
        "detail": (
            "Build a Hindi Soundex table that maps similar-sounding matra "
            "combinations (e.g., ा/आ, ि/ी, े/ै) to canonical forms. "
            "After Whisper decodes, run a second pass using a trigram "
            "language model trained on clean Hindi text to pick the "
            "most probable matra variant. Implementation: create a "
            "confusion matrix of matra substitutions from training errors, "
            "then use it as a weighted FST (Finite State Transducer) "
            "for re-scoring."
        ),
    },
    "Conjunct Consonant Error": {
        "fix": "Custom BPE Vocabulary with Conjunct-Preserving Tokenization",
        "detail": (
            "Extend the Whisper tokenizer with high-frequency Hindi "
            "conjuncts (क्ष, त्र, ज्ञ, श्र, etc.) as single tokens. "
            "Re-train the tokenizer on the Hindi corpus, ensuring "
            "halant+consonant sequences are kept intact. This prevents "
            "the model from splitting conjuncts into separate characters "
            "during decoding. Then fine-tune the decoder head for 500 "
            "additional steps on conjunct-heavy samples."
        ),
    },
    "Number/Digit Handling": {
        "fix": "Rule-Based Number Normalizer Post-Processor",
        "detail": (
            "Implement a regex pipeline that: (1) Detects Hindi number "
            "words (एक, दो, तीन...) and converts to digits; "
            "(2) Detects English digits and converts to Devanagari; "
            "(3) Handles ordinals (पहला→1st) and large numbers "
            "(दो लाख→2,00,000). Use the `indic_numtowords` library "
            "for bidirectional conversion. Run this as a deterministic "
            "post-processing step after Whisper output."
        ),
    },
    "English Loanword Error": {
        "fix": "Bilingual Lexicon-Constrained Decoding",
        "detail": (
            "Create a bilingual glossary of ~500 common English loanwords "
            "used in Hindi (with Devanagari spellings). During beam search "
            "decoding, boost the log-probability of tokens that form known "
            "loanword spellings using `logits_processor` in Whisper's "
            "generate(). This acts as soft lexicon-constrained decoding "
            "without hard-forcing, preserving model flexibility."
        ),
    },
    "Short Word Insertion/Deletion": {
        "fix": "Attention-Weighted Silence Detection",
        "detail": (
            "Short words (है, का, में, से, को) are often inserted or "
            "deleted due to poor silence/boundary detection. Add a "
            "pre-processing step using VAD (Voice Activity Detection) "
            "with 50ms frame resolution to mark word boundaries. Then "
            "train a small CTC aligner on the fine-tuning data to "
            "produce forced alignments, filtering out low-confidence "
            "short word insertions below a threshold score."
        ),
    },
    "Word Substitution": {
        "fix": "Context-Aware Re-Scoring with Hindi LM",
        "detail": (
            "Train a 5-gram KenLM language model on a large Hindi text "
            "corpus (e.g., Hindi Wikipedia + CC-100). Use shallow fusion "
            "during Whisper beam search: combine Whisper's token scores "
            "with LM scores using interpolation weight λ=0.3. This "
            "corrects phonetically similar but contextually wrong word "
            "substitutions. Use `pyctcdecode` for efficient integration."
        ),
    },
    "Repetition/Hallucination": {
        "fix": "N-gram Repetition Penalty + Length Normalization",
        "detail": (
            "Apply a no-repeat-ngram penalty during Whisper's beam "
            "search decoding by setting `no_repeat_ngram_size=3` in "
            "model.generate(). Additionally, use length normalization "
            "(length_penalty=1.0) to prevent the decoder from generating "
            "excessively long outputs. As a post-processing step, "
            "detect consecutive duplicate trigrams and collapse them."
        ),
    },
    "Proper Noun Confusion": {
        "fix": "Hot-Word Biasing with Domain-Specific Entity List",
        "detail": (
            "Compile a list of proper nouns from the training data "
            "(names, places, organizations). During decoding, use "
            "Whisper's `prompt` or `prefix` feature to inject likely "
            "entity names as decoder prompts. For production, maintain "
            "a trie of known entities and boost matching token sequences "
            "during beam search."
        ),
    },
    "Other/Phonetic Similarity": {
        "fix": "Phonetic Similarity Re-Ranking",
        "detail": (
            "Use n-best list re-ranking: generate top-10 hypotheses "
            "from Whisper, then re-rank using phonetic edit distance "
            "(based on Hindi phoneme inventory) combined with a language "
            "model score. This catches cases where the correct word is "
            "phonetically close to the predicted word."
        ),
    },
}


def task3_actionable_fixes(all_categories: Counter):
    """Print top 3 error types and their specific fixes."""
    print("\n" + "=" * 80)
    print("  TASK 3: ACTIONABLE FIXES FOR TOP 3 ERROR TYPES")
    
    top3 = all_categories.most_common(3)

    for rank, (cat, count) in enumerate(top3, 1):
        fix_info = FIXES.get(cat, {
            "fix": "Data Augmentation + Targeted Fine-Tuning",
            "detail": (
                "Collect more examples of this error pattern, augment "
                "using speed/pitch perturbation, and fine-tune Whisper "
                "for additional steps with a higher weight on these samples."
            ),
        })

        print(f"\n  ╔═ #{rank} — {cat} ({count} occurrences)")
        print(f"  ║")
        print(f"  ║  FIX: {fix_info['fix']}")
        print(f"  ║")
        # Word-wrap the detail
        detail = fix_info["detail"]
        words = detail.split()
        line = "  ║  "
        for word in words:
            if len(line) + len(word) > 75:
                print(line)
                line = "  ║  " + word + " "
            else:
                line += word + " "
        if line.strip("║ "):
            print(line)
        print(f"  ║")
        print(f"  ╚{'═' * 70}")

    # Save fixes summary
    fixes_rows = []
    for rank, (cat, count) in enumerate(top3, 1):
        fix_info = FIXES.get(cat, {"fix": "N/A", "detail": "N/A"})
        fixes_rows.append({
            "rank": rank,
            "error_category": cat,
            "occurrences": count,
            "proposed_fix": fix_info["fix"],
            "implementation_detail": fix_info["detail"],
        })
    fixes_df = pd.DataFrame(fixes_rows)
    out_path = BASE_DIR / "actionable_fixes.csv"
    fixes_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  → Fixes saved to: {out_path}")


#  MAIN

def main():
    parser = argparse.ArgumentParser(description="Hindi ASR Error Analysis")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV,
                        help="Path to predictions CSV (reference, prediction, wer_percent)")
    args = parser.parse_args()

        print("  HINDI ASR ERROR ANALYSIS")
        print(f"  Input: {args.csv}")

    # Load data
    df = pd.read_csv(args.csv)

    # Handle column name variants
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ("reference", "ref", "ground_truth"):
            col_map[col] = "reference"
        elif cl in ("prediction", "hypothesis", "hyp", "pred"):
            col_map[col] = "prediction"
        elif cl in ("wer_percent", "wer_score", "wer"):
            col_map[col] = "wer_percent"
    df = df.rename(columns=col_map)

    # Compute WER if not present
    if "wer_percent" not in df.columns:
        print("  Computing WER per sample …")
        wers = []
        for _, row in df.iterrows():
            try:
                w = jiwer.wer(str(row["reference"]), str(row["prediction"])) * 100
            except ValueError:
                w = 100.0
            wers.append(w)
        df["wer_percent"] = wers

    print(f"  Loaded {len(df)} samples")
    print(f"  Overall WER: {df['wer_percent'].mean():.1f}%")
    print(f"  WER range  : {df['wer_percent'].min():.1f}% — {df['wer_percent'].max():.1f}%")

    # Run all 3 tasks
    sampled = task1_systematic_sampling(df)
    all_categories, _ = task2_error_taxonomy(sampled)
    task3_actionable_fixes(all_categories)

    print("\n" + "=" * 80)
    print("  ANALYSIS COMPLETE")
        print(f"\n  Output files:")
    print(f"    • error_samples_25.csv    — 25 sampled examples")
    print(f"    • error_taxonomy.csv      — categorized errors")
    print(f"    • actionable_fixes.csv    — top 3 fixes")
    print()


if __name__ == "__main__":
    main()
