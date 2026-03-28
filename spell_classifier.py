import re
import os
import argparse
import requests
import pandas as pd
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).parent
DICT_DIR = BASE_DIR / "hindi_dictionaries"

# Reference Dictionary

DICTIONARY_URLS = [
    # Main Hindi wordlists from GitHub
    "https://raw.githubusercontent.com/pradeepbishnoi/hinglish/master/dictionary.txt",
    "https://raw.githubusercontent.com/mishrasunny174/WordLists/master/HindiWordlist.txt",
    # Hunspell Hindi dictionary (large)
    "https://raw.githubusercontent.com/Shreeshrii/hindi-hunspell/master/Hindi/hi_IN.dic",
    # Indic Keyboard wordlist
    "https://raw.githubusercontent.com/nicedoc/wordlists/main/hi.txt",
]


def _download_file(url: str, dest: Path) -> bool:
    """Download a file if it doesn't already exist."""
    if dest.exists() and dest.stat().st_size > 100:
        return True
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except Exception as e:
        print(f"    ⚠ Failed: {url.split('/')[-1]}: {e}")
        return False


def _extract_words_from_file(path: Path) -> set[str]:
    """Extract Devanagari words from a text file (handles hunspell .dic format)."""
    words = set()
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return words

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("%"):
            continue
        # Hunspell .dic: word/flags → take just the word part
        word = line.split("/")[0].strip()
        # Skip lines that are just numbers (hunspell header)
        if word.isdigit():
            continue
        if word and _is_devanagari(word):
            words.add(word)
    return words


def load_dictionary() -> set[str]:
    """Load Hindi dictionary from multiple sources."""
    print("\nLoading reference dictionaries …")
    DICT_DIR.mkdir(exist_ok=True)

    all_words = set()

    # Download external dictionaries
    for i, url in enumerate(DICTIONARY_URLS):
        fname = DICT_DIR / f"dict_{i}.txt"
        if _download_file(url, fname):
            words = _extract_words_from_file(fname)
            print(f"    Source {i}: {len(words):>6,} words  [{url.split('/')[-1]}]")
            all_words.update(words)

    # Also load any .txt files user may have placed in hindi_dictionaries/
    for txt_file in DICT_DIR.glob("*.txt"):
        if txt_file.name.startswith("dict_"):
            continue  # already loaded above
        words = _extract_words_from_file(txt_file)
        if words:
            print(f"    Custom : {len(words):>6,} words  [{txt_file.name}]")
            all_words.update(words)

    #  Large built-in core vocabulary (~500 words) 
    BUILTIN = set()
    
    # Pronouns (all forms)
    BUILTIN.update("मैं तू तुम आप वह वे यह ये हम वो इन्हें उन्हें जिन्हें "
                   "उसका उसकी उसके मेरा मेरी मेरे तेरा तेरी तेरे तुम्हारा तुम्हारी तुम्हारे "
                   "हमारा हमारी हमारे उनका उनकी उनके इसका इसकी इसके इनका इनकी इनके "
                   "कोई कुछ सब अपना अपनी अपने खुद स्वयं जो कौन क्या किसी".split())

    # Postpositions & particles
    BUILTIN.update("में पर को से के का की ने तक द्वारा "
                   "लिए लिये बाद पहले ऊपर नीचे बीच साथ बिना अंदर बाहर सामने पीछे बारे "
                   "अनुसार अलावा बजाय विरुद्ध प्रति दौरान बावजूद".split())

    # Conjunctions & connectors
    BUILTIN.update("और या पर लेकिन मगर परंतु किंतु परन्तु किन्तु "
                   "क्योंकि इसलिए जब तब अगर तो कि ताकि जबकि हालांकि "
                   "चाहे बल्कि यानी अर्थात यानि एवं तथा अथवा".split())

    # Copula / auxiliary verbs (all tenses)
    BUILTIN.update("है हैं था थी थे थीं हूँ हूं हो होगा होगी होंगे होंगी "
                   "होता होती होते होना हुआ हुई हुए हुईं".split())

    # Common verbs: root + all frequent inflections
    verb_roots = [
        "कर", "हो", "जा", "आ", "दे", "ले", "कह", "बोल",
        "सुन", "देख", "पढ़", "लिख", "खा", "पी", "सो", "रह",
        "चल", "बैठ", "मिल", "रख", "सोच", "समझ", "चाह",
        "पा", "लग", "मान", "बता", "जान", "मार", "उठ",
        "बन", "बना", "खेल", "गा", "रो", "हँस",
        "पहुँच", "निकल", "पूछ", "भेज", "खोल", "बंद",
        "सीख", "भूल", "रुक", "गिर", "उड़", "तोड़",
        "छोड़", "पकड़", "काट", "धो", "चुन", "बदल",
        "डाल", "निकाल", "हटा", "बचा", "जोड़",
        "सम्भाल", "संभाल", "इस्तेमाल",
    ]
    verb_suffixes = [
        "ना", "ता", "ती", "ते", "ा", "ी", "े",
        "ो", "ें", "ूँ", "ूं", "ेगा", "ेगी", "ेंगे", "ेंगी",
        "ाना", "ाती", "ाता", "ाते", "ाया", "ाई", "ाए",
        "ाओ", "ाये", "ाइए", "ाइये",
    ]
    for root in verb_roots:
        BUILTIN.add(root)
        for suf in verb_suffixes:
            BUILTIN.add(root + suf)

    # Common adjectives (all gender forms)
    adj_triples = [
        ("अच्छा", "अच्छी", "अच्छे"), ("बुरा", "बुरी", "बुरे"),
        ("बड़ा", "बड़ी", "बड़े"), ("छोटा", "छोटी", "छोटे"),
        ("नया", "नई", "नए"), ("पुराना", "पुरानी", "पुराने"),
        ("पूरा", "पूरी", "पूरे"), ("काला", "काली", "काले"),
        ("सफ़ेद", "सफ़ेद", "सफ़ेद"), ("लाल", "लाल", "लाल"),
        ("ऊँचा", "ऊँची", "ऊँचे"), ("लंबा", "लंबी", "लंबे"),
        ("गहरा", "गहरी", "गहरे"), ("मोटा", "मोटी", "मोटे"),
        ("पतला", "पतली", "पतले"), ("गरम", "गरम", "गरम"),
        ("ठंडा", "ठंडी", "ठंडे"), ("सस्ता", "सस्ती", "सस्ते"),
        ("महंगा", "महंगी", "महंगे"),
    ]
    for forms in adj_triples:
        BUILTIN.update(forms)
    BUILTIN.update("सही गलत ग़लत ज़रूरी ज़्यादा ज्यादा कम काफ़ी काफी "
                   "बहुत थोड़ा थोड़ी थोड़े सारा सारी खास अलग पहला दूसरा तीसरा "
                   "ख़ास ज़रा ज़रूर ज़ोर".split())

    # Common nouns
    BUILTIN.update("लोग बात बातें काम दिन रात समय वक़्त वक्त "
                   "साल महीना हफ़्ता हफ्ता घंटा मिनट तरह तरीका "
                   "घर दरवाज़ा दरवाजा कमरा रास्ता जगह शहर गाँव गांव "
                   "देश दुनिया ज़मीन ज़िंदगी जिंदगी ज़िन्दगी जिन्दगी "
                   "आसमान पानी हवा आग पेड़ फूल फल पत्ता नदी पहाड़ समुद्र "
                   "आदमी औरत बच्चा बच्ची बच्चे लड़का लड़की माँ पिता बाप "
                   "भाई बहन बहिन दोस्त परिवार भगवान ईश्वर धर्म पूजा मंदिर "
                   "सरकार नेता मंत्री राजा प्रधानमंत्री पैसा रुपया रुपये "
                   "खाना दूध चाय रोटी चावल दाल सब्ज़ी सब्जी नमक चीनी "
                   "कपड़ा कपड़े किताब क़लम कलम कागज़ कागज गाड़ी सड़क पुल "
                   "आँख कान नाक मुँह हाथ पैर सिर पेट दिल दिमाग़ दिमाग "
                   "आवाज़ आवाज ख़बर खबर ख़ुशी खुशी मुश्किल "
                   "तस्वीर इतिहास भविष्य वर्तमान".split())

    # Adverbs / discourse particles
    BUILTIN.update("अब तब कब जब यहाँ यहां वहाँ वहां कहाँ कहां जहाँ जहां "
                   "आज कल परसों अभी हमेशा कभी फिर भी ही तो न ना नहीं मत "
                   "हाँ हां जी बस काश शायद ज़रूर ज़रा सच सिर्फ़ सिर्फ "
                   "केवल बिल्कुल ठीक वैसे ऐसे कैसे ऐसा वैसा कैसा "
                   "इतना उतना जितना कितना सबसे बहुत ज़्यादा कम "
                   "तुरंत जल्दी धीरे अचानक ज़रूर बार".split())

    # Numbers
    BUILTIN.update("एक दो तीन चार पाँच पांच छह छः सात आठ नौ दस "
                   "ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस बीस "
                   "सौ हज़ार हजार लाख करोड़".split())

    # Honorifics / titles
    BUILTIN.update("श्री श्रीमती सुश्री महोदय महोदया जनाब साहब".split())

    all_words.update(BUILTIN)

    # Import English loanwords (these are CORRECT)
    try:
        from post_processing import ENGLISH_LOANWORDS, ABSORBED_HINDI_WORDS
        all_words.update(ENGLISH_LOANWORDS.keys())
        all_words.update(ABSORBED_HINDI_WORDS)
    except ImportError:
        pass

    print(f"    ────────────────────────────────")
    print(f"    Total dictionary: {len(all_words):,} unique words")
    return all_words


# Phonotactic Rules

DEVANAGARI_RANGE = re.compile(r"[\u0900-\u097F]")
VOWELS = set("अआइईउऊऋएऐओऔ")
VOWEL_SIGNS = set("ािीुूृेैोौॅॉ")
NASALS = set("ंँः")  # anusvara, chandrabindu, visarga
CONSONANTS = set("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह")
HALANT = "्"

# Patterns for HIGH confidence INCORRECT
DOUBLE_MATRA = re.compile(r"[ािीुूृेैोौ]{2,}")   # ाा, ीी etc.
TRIPLE_REPEAT = re.compile(r"(.)\1{2,}")            # aaa
MIXED_SCRIPT = re.compile(r"[\u0900-\u097F].*[a-zA-Z]|[a-zA-Z].*[\u0900-\u097F]")

# Valid Hindi word structure regex (consonant clusters, vowels, matras, nasals)
VALID_DEVANAGARI_WORD = re.compile(
    r"^[\u0900-\u097F\u200C\u200D़]+$"  # only Devanagari + nukta + ZWNJ/ZWJ
)

# Common valid suffixes for morphological analysis
VALID_SUFFIXES = [
    # Verb inflections
    "ता", "ती", "ते", "ना", "नी", "ने", "या", "यी", "ये",
    "ेगा", "ेगी", "ेंगे", "ेंगी", "ाना", "ाई", "ाए", "ाओ",
    # Noun / adjective suffixes
    "वाला", "वाली", "वाले", "कार", "दार", "गार",
    "पन", "पना", "आई", "ाई", "ीय", "इक", "ीन", "आत",
    "ता", "त्व", "ित", "मान", "वान",
    # Plurals
    "ें", "ों", "ियाँ", "ियां", "ओं",
    # Oblique
    "ो", "ी", "े",
]


def _is_devanagari(word: str) -> bool:
    """Check if a word contains primarily Devanagari characters."""
    dev_chars = sum(1 for c in word if DEVANAGARI_RANGE.match(c))
    return dev_chars > len(word) * 0.5


def _is_pure_devanagari(word: str) -> bool:
    """Check if word contains ONLY Devanagari characters (+ nukta/virama)."""
    return bool(VALID_DEVANAGARI_WORD.match(word))


def _has_valid_structure(word: str) -> tuple[bool, str]:
    """Check phonotactic validity. Returns (is_valid, reason)."""
    if not word:
        return False, "Empty word"

    # HIGH INCORRECT: Mixed Roman + Devanagari
    if MIXED_SCRIPT.search(word):
        return False, "Invalid: mixed Roman + Devanagari script"

    # HIGH INCORRECT: Double matras (ाा, ीी, etc.)
    if DOUBLE_MATRA.search(word):
        return False, "Invalid: repeated matra (ाा / ीी / etc.)"

    # HIGH INCORRECT: Triple repetition (e.g., ककक)
    if TRIPLE_REPEAT.search(word):
        return False, "Invalid: triple character repetition"

    # HIGH INCORRECT: Starts with a vowel sign (matra without consonant)
    if word[0] in VOWEL_SIGNS or word[0] in NASALS:
        return False, "Invalid: starts with vowel sign / nasal"

    # SUSPICIOUS: Ends with halant (incomplete conjunct)
    if word.endswith(HALANT):
        return False, "Suspicious: ends with halant"

    # Invalid: vowel sign directly after independent vowel
    for i in range(1, len(word)):
        if word[i] in VOWEL_SIGNS and word[i - 1] in VOWELS:
            return False, "Invalid: vowel sign after independent vowel"

    # Check for only non-Devanagari content
    if not any(DEVANAGARI_RANGE.match(c) for c in word):
        return False, "Not Devanagari"

    return True, "Valid"


# Classification

def classify_word(word: str, dictionary: set[str]) -> dict:
    """
    Classify a single word.
    
    Priority order:
      1. SKIP: Not Devanagari
      2. HIGH CORRECT: In dictionary
      3. HIGH INCORRECT: Structural violations
      4. MEDIUM CORRECT: Valid structure + morphological match
      5. MEDIUM CORRECT: Valid structure, 2-15 chars, pure Devanagari
      6. LOW: Ambiguous (very short / very long / unusual structure)
    """
    word = str(word).strip()

    #  SKIP: not Devanagari 
    if not word or not _is_devanagari(word):
        return {
            "word": word,
            "classification": "SKIP",
            "confidence": "N/A",
            "reason": "Not Devanagari text",
        }

    #  HIGH CORRECT: exact dictionary match 
    if word in dictionary:
        return {
            "word": word,
            "classification": "CORRECT",
            "confidence": "HIGH",
            "reason": "Found in reference dictionary",
        }

    #  Check structural validity 
    is_valid, reason = _has_valid_structure(word)

    if not is_valid:
        #  HIGH INCORRECT: structural violation 
        return {
            "word": word,
            "classification": "INCORRECT",
            "confidence": "HIGH",
            "reason": reason,
        }

    #  From here: word is NOT in dictionary but IS phonotactically valid 

    # Normalize: remove nukta / chandrabindu variants → check dict again
    normalized = word.replace("़", "").replace("ँ", "ं").replace("ॉ", "ो").replace("ॅ", "े")
    if normalized != word and normalized in dictionary:
        return {
            "word": word,
            "classification": "CORRECT",
            "confidence": "HIGH",
            "reason": f"Nukta/nasal variant of '{normalized}'",
        }

    #  MEDIUM CORRECT: root + valid suffix → morphological match 
    for suffix in VALID_SUFFIXES:
        if len(suffix) < len(word) and word.endswith(suffix):
            root = word[: -len(suffix)]
            if root in dictionary:
                return {
                    "word": word,
                    "classification": "CORRECT",
                    "confidence": "MEDIUM",
                    "reason": f"Root '{root}' + suffix '{suffix}'",
                }
            # Also check normalized root
            norm_root = root.replace("़", "").replace("ँ", "ं")
            if norm_root != root and norm_root in dictionary:
                return {
                    "word": word,
                    "classification": "CORRECT",
                    "confidence": "MEDIUM",
                    "reason": f"Root variant '{norm_root}' + suffix '{suffix}'",
                }

    #  Single valid Devanagari character → HIGH CORRECT 
    # (postpositions like है, तो, में are single/double chars)
    if len(word) <= 2 and _is_pure_devanagari(word):
        # Very short and not in dictionary — could be valid particle or not
        # Single consonant/vowel letters are valid
        if word[0] in CONSONANTS | VOWELS:
            return {
                "word": word,
                "classification": "CORRECT",
                "confidence": "MEDIUM",
                "reason": "Short valid Devanagari word (likely particle/suffix)",
            }

    #  MEDIUM CORRECT: pure Devanagari, valid structure, reasonable length 
    if _is_pure_devanagari(word) and 2 <= len(word) <= 15:
        return {
            "word": word,
            "classification": "CORRECT",
            "confidence": "MEDIUM",
            "reason": "Valid Devanagari structure (2-15 chars, no violations)",
        }

    #  Words > 15 chars: likely compound or error 
    if len(word) > 15 and _is_pure_devanagari(word):
        return {
            "word": word,
            "classification": "INCORRECT",
            "confidence": "LOW",
            "reason": "Unusually long word (>15 chars) — possible concatenation error",
        }

    #  Default: LOW confidence 
    return {
        "word": word,
        "classification": "INCORRECT",
        "confidence": "LOW",
        "reason": "Not in dictionary, ambiguous structure — needs manual review",
    }


# Pipeline

def run_pipeline(input_csv: str, output_csv: str):
    """Run the full classification pipeline."""

    print("  HINDI SPELLING CLASSIFICATION PIPELINE  v2")
    
    # Load words
    print(f"\n  Input: {input_csv}")
    df = pd.read_csv(input_csv)
    col = df.columns[0]
    words = df[col].dropna().astype(str).unique().tolist()
    print(f"  Unique words: {len(words):,}")

    # Load dictionary
    dictionary = load_dictionary()

    # Classify
    print("\nClassifying words …")
    results = []
    for i, word in enumerate(words):
        results.append(classify_word(word, dictionary))
        if (i + 1) % 50000 == 0:
            print(f"    Processed {i + 1:,} / {len(words):,} …")

    results_df = pd.DataFrame(results)
    print(f"    Done: {len(results):,} words classified")

    #  Statistics 
    print("\n" + "=" * 70)
    print("  CLASSIFICATION RESULTS")
    
    class_counts = results_df["classification"].value_counts()
    print(f"\n  Classification:")
    for cls, count in class_counts.items():
        pct = count / len(results_df) * 100
        bar = "█" * int(pct / 2)
        print(f"    {cls:<12} {count:>8,}  ({pct:>5.1f}%)  {bar}")

    print(f"\n  Confidence:")
    conf_counts = results_df["confidence"].value_counts()
    for conf, count in conf_counts.items():
        pct = count / len(results_df) * 100
        bar = "▓" * int(pct / 2)
        print(f"    {conf:<12} {count:>8,}  ({pct:>5.1f}%)  {bar}")

    print(f"\n  Classification × Confidence:")
    cross = pd.crosstab(results_df["classification"], results_df["confidence"])
    print("    " + cross.to_string().replace("\n", "\n    "))

    print(f"\n  Top Reasons:")
    for reason, count in results_df["reason"].value_counts().head(10).items():
        print(f"    {count:>8,}  {reason[:70]}")

    #  Save outputs 
    out_path = Path(output_csv)
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  → Full results: {out_path}")

    low_conf = results_df[results_df["confidence"] == "LOW"]
    review_sample = low_conf.sample(n=min(50, len(low_conf)), random_state=42)
    review_path = BASE_DIR / "manual_review_words.csv"
    review_sample.to_csv(review_path, index=False, encoding="utf-8-sig")
    print(f"  → {len(review_sample)} words for manual review: {review_path}")

    high_incorrect = results_df[
        (results_df["classification"] == "INCORRECT") &
        (results_df["confidence"] == "HIGH")
    ]
    hi_path = BASE_DIR / "high_confidence_incorrect.csv"
    high_incorrect.to_csv(hi_path, index=False, encoding="utf-8-sig")
    print(f"  → {len(high_incorrect):,} high-confidence incorrect: {hi_path}")

    print(f"\n  Summary:")
    print(f"    Total words      : {len(results_df):,}")
    print(f"    CORRECT          : {class_counts.get('CORRECT', 0):,}")
    print(f"    INCORRECT        : {class_counts.get('INCORRECT', 0):,}")
    print(f"    SKIP             : {class_counts.get('SKIP', 0):,}")
    low_pct = len(low_conf) / len(results_df) * 100
    print(f"    LOW confidence   : {len(low_conf):,} ({low_pct:.1f}%)")
    print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Hindi Spelling Classifier v2")
    parser.add_argument("--input", type=str,
                        default=str(BASE_DIR / "Unique Words Data - Sheet1.csv"))
    parser.add_argument("--output", type=str,
                        default=str(BASE_DIR / "spelling_classification.csv"))
    args = parser.parse_args()
    run_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()
