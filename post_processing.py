import re
import argparse
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent


#  PART A — Hindi Number Word → Digit Converter

#  Comprehensive mapping: Hindi number words (0–99) 

UNITS = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4,
    "पांच": 5, "पाँच": 5, "छह": 6, "छः": 6, "सात": 7,
    "आठ": 8, "नौ": 9,
}

TEENS_AND_TENS = {
    "दस": 10, "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14,
    "पंद्रह": 15, "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19,
    "बीस": 20, "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24,
    "पच्चीस": 25, "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28, "उनतीस": 29,
    "तीस": 30, "इकतीस": 31, "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34,
    "पैंतीस": 35, "छत्तीस": 36, "सैंतीस": 37, "अड़तीस": 38, "उनतालीस": 39,
    "चालीस": 40, "इकतालीस": 41, "बयालीस": 42, "तैंतालीस": 43, "चवालीस": 44,
    "पैंतालीस": 45, "छियालीस": 46, "सैंतालीस": 47, "अड़तालीस": 48, "उनचास": 49,
    "पचास": 50, "इक्यावन": 51, "बावन": 52, "तिरपन": 53, "चौवन": 54,
    "पचपन": 55, "छप्पन": 56, "सत्तावन": 57, "अट्ठावन": 58, "उनसठ": 59,
    "साठ": 60, "इकसठ": 61, "बासठ": 62, "तिरसठ": 63, "चौंसठ": 64,
    "पैंसठ": 65, "छियासठ": 66, "सड़सठ": 67, "अड़सठ": 68, "उनहत्तर": 69,
    "सत्तर": 70, "इकहत्तर": 71, "बहत्तर": 72, "तिहत्तर": 73, "चौहत्तर": 74,
    "पचहत्तर": 75, "छिहत्तर": 76, "सतहत्तर": 77, "अठहत्तर": 78, "उनासी": 79,
    "अस्सी": 80, "इक्यासी": 81, "बयासी": 82, "तिरासी": 83, "चौरासी": 84,
    "पचासी": 85, "छियासी": 86, "सत्तासी": 87, "अट्ठासी": 88, "नवासी": 89,
    "नब्बे": 90, "इक्यानवे": 91, "बानवे": 92, "तिरानवे": 93, "चौरानवे": 94,
    "पचानवे": 95, "छियानवे": 96, "सत्तानवे": 97, "अट्ठानवे": 98, "निन्यानवे": 99,
}

MULTIPLIERS = {
    "सौ": 100,
    "हज़ार": 1000, "हजार": 1000, "हज़ार": 1000,
    "लाख": 100000, "लाख़": 100000,
    "करोड़": 10000000, "करोड": 10000000,
}

# Combined lookup for all number words (0–99)
ALL_NUMBER_WORDS = {}
ALL_NUMBER_WORDS.update(UNITS)
ALL_NUMBER_WORDS.update(TEENS_AND_TENS)

#  Idiomatic patterns to skip 

IDIOMATIC_PATTERNS = [
    re.compile(r"दो-चार\s+\S+"),          # दो-चार बातें
    re.compile(r"दो-तीन\s+\S+"),           # दो-तीन लोग
    re.compile(r"चार-पांच\s+\S+"),          # चार-पांच दिन
    re.compile(r"एक-दो\s+\S+"),            # एक-दो बार
    re.compile(r"दो-एक\s+\S+"),            # दो-एक मिनट
    re.compile(r"तीन-चार\s+\S+"),           # तीन-चार घंटे
    re.compile(r"दस-बीस\s+\S+"),           # दस-बीस रुपये
    re.compile(r"सौ-दो\s+सौ"),             # सौ-दो सौ
    re.compile(r"एकाध\s+\S+"),             # एकाध बार
    re.compile(r"चारों\s+"),               # चारों तरफ
    re.compile(r"दोनों\s+"),               # दोनों लोग
    re.compile(r"तीनों\s+"),               # तीनों भाई
    re.compile(r"पहला|दूसरा|तीसरा|चौथा"),  # ordinals — keep as-is
]


def _is_idiomatic(text: str, match_start: int, match_end: int) -> bool:
    """Check if a number word at the given position is part of an idiom."""
    # Check a window around the match
    window_start = max(0, match_start - 10)
    window_end = min(len(text), match_end + 30)
    window = text[window_start:window_end]

    for pattern in IDIOMATIC_PATTERNS:
        if pattern.search(window):
            return True
    return False


def _parse_number_sequence(words: list[str]) -> tuple[int, int]:
    """
    Parse a sequence of Hindi number words into a single integer.
    Returns (value, number_of_words_consumed).

    Handles: तीन सौ चौवन → 354, एक लाख → 100000, etc.

    Grammar:
        number = (small_num multiplier)* small_num?
        small_num = 0–99 from lookup
        multiplier = सौ | हज़ार | लाख | करोड़
    """
    total = 0
    current = 0
    consumed = 0

    i = 0
    while i < len(words):
        word = words[i]

        if word in ALL_NUMBER_WORDS:
            current = ALL_NUMBER_WORDS[word]
            consumed = i + 1
            i += 1
        elif word in MULTIPLIERS:
            mult = MULTIPLIERS[word]
            if current == 0:
                current = 1  # "सौ" alone = 100
            if mult > total:
                # e.g., "दो लाख तीन हज़ार" → (2*100000) + ...
                total = (total + current) * mult
            else:
                # e.g., "तीन सौ" → current * mult added to total
                total += current * mult
            current = 0
            consumed = i + 1
            i += 1
        else:
            break  # Non-number word → stop

    total += current
    return total, consumed


def convert_hindi_numbers(text: str) -> str:
    """
    Convert Hindi number words to digits in a string.
    Skips idiomatic expressions.
    """
    words = text.split()
    result = []
    i = 0

    while i < len(words):
        word = words[i]

        # Check if this word is a number word or multiplier
        if word in ALL_NUMBER_WORDS or word in MULTIPLIERS:
            # Check for idiomatic usage
            char_pos = len(" ".join(words[:i]))
            match_end = char_pos + len(word)
            if _is_idiomatic(text, char_pos, match_end):
                result.append(word)
                i += 1
                continue

            # Try to parse a multi-word number
            value, consumed = _parse_number_sequence(words[i:])
            if consumed > 0 and value > 0:
                result.append(str(value))
                i += consumed
            else:
                result.append(word)
                i += 1
        else:
            result.append(word)
            i += 1

    return " ".join(result)


#  PART B — English Loanword Tagger

# 200+ common English loanwords written in Devanagari
ENGLISH_LOANWORDS = {
    #  Technology 
    "कंप्यूटर": "computer", "मोबाइल": "mobile", "फोन": "phone",
    "इंटरनेट": "internet", "ऑनलाइन": "online", "ऑफलाइन": "offline",
    "सॉफ्टवेयर": "software", "हार्डवेयर": "hardware", "वेबसाइट": "website",
    "ईमेल": "email", "पासवर्ड": "password", "डाउनलोड": "download",
    "अपलोड": "upload", "अपडेट": "update", "डिलीट": "delete",
    "सर्वर": "server", "डेटा": "data", "डेटाबेस": "database",
    "नेटवर्क": "network", "वाईफाई": "wifi", "ब्लूटूथ": "bluetooth",
    "एप्लिकेशन": "application", "ऐप": "app", "ब्राउज़र": "browser",
    "गूगल": "google", "लैपटॉप": "laptop", "टैबलेट": "tablet",
    "सोशल": "social", "मीडिया": "media", "डिजिटल": "digital",
    "टेक्नोलॉजी": "technology", "प्रोग्राम": "program",
    "कोड": "code", "प्रोग्रामिंग": "programming",
    "एल्गोरिदम": "algorithm", "इंजन": "engine", "सिस्टम": "system",

    #  Education 
    "स्कूल": "school", "कॉलेज": "college", "यूनिवर्सिटी": "university",
    "टीचर": "teacher", "स्टूडेंट": "student", "क्लास": "class",
    "एग्जाम": "exam", "रिजल्ट": "result", "डिग्री": "degree",
    "कोर्स": "course", "सर्टिफिकेट": "certificate",
    "ट्यूशन": "tuition", "लाइब्रेरी": "library",
    "लेक्चर": "lecture", "प्रोफेसर": "professor",
    "सेमेस्टर": "semester", "सब्जेक्ट": "subject",

    #  Business / Work 
    "बिजनेस": "business", "कंपनी": "company", "ऑफिस": "office",
    "मीटिंग": "meeting", "मैनेजर": "manager", "बॉस": "boss",
    "प्रोजेक्ट": "project", "रिपोर्ट": "report", "प्रेजेंटेशन": "presentation",
    "इंटरव्यू": "interview", "जॉब": "job", "सैलरी": "salary",
    "प्रमोशन": "promotion", "रिज्यूम": "resume",
    "क्लाइंट": "client", "कस्टमर": "customer",
    "मार्केट": "market", "मार्केटिंग": "marketing",
    "प्रोडक्ट": "product", "सर्विस": "service",
    "ब्रांड": "brand", "एडवर्टाइजिंग": "advertising",
    "इन्वेस्टमेंट": "investment", "बजट": "budget",
    "प्रॉफिट": "profit", "लॉस": "loss",
    "टार्गेट": "target", "स्ट्रेटेजी": "strategy",
    "पार्टनर": "partner", "डील": "deal",

    #  Health 
    "हॉस्पिटल": "hospital", "डॉक्टर": "doctor", "नर्स": "nurse",
    "पेशेंट": "patient", "मेडिसिन": "medicine", "अपॉइंटमेंट": "appointment",
    "रिपोर्ट": "report", "टेस्ट": "test", "ऑपरेशन": "operation",
    "इंजेक्शन": "injection", "एंटीबायोटिक": "antibiotic",
    "वैक्सीन": "vaccine", "इम्यूनिटी": "immunity",
    "सर्जरी": "surgery", "थेरेपी": "therapy",
    "फार्मेसी": "pharmacy", "सप्लीमेंट": "supplement",

    #  Transport / Travel 
    "बस": "bus", "ट्रेन": "train", "टैक्सी": "taxi",
    "एयरपोर्ट": "airport", "स्टेशन": "station", "टिकट": "ticket",
    "पासपोर्ट": "passport", "विज़ा": "visa", "होटल": "hotel",
    "फ्लाइट": "flight", "ड्राइवर": "driver", "पार्किंग": "parking",
    "हाईवे": "highway", "रूट": "route", "ट्रैफिक": "traffic",
    "पेट्रोल": "petrol", "डीज़ल": "diesel",

    #  Food / Daily Life 
    "रेस्टोरेंट": "restaurant", "पिज़्ज़ा": "pizza", "बर्गर": "burger",
    "चॉकलेट": "chocolate", "कॉफी": "coffee", "बिस्किट": "biscuit",
    "केक": "cake", "आइसक्रीम": "icecream", "ज्यूस": "juice",
    "सॉस": "sauce", "सलाद": "salad", "सैंडविच": "sandwich",
    "नूडल्स": "noodles", "मैगी": "maggi",

    #  Sports 
    "क्रिकेट": "cricket", "फुटबॉल": "football", "हॉकी": "hockey",
    "टेनिस": "tennis", "बैडमिंटन": "badminton", "ओलंपिक": "olympics",
    "मैच": "match", "टीम": "team", "प्लेयर": "player",
    "कोच": "coach", "स्कोर": "score", "गोल": "goal",
    "ट्रॉफी": "trophy", "चैंपियन": "champion",
    "स्टेडियम": "stadium", "कप्तान": "captain",

    #  Entertainment 
    "फिल्म": "film", "मूवी": "movie", "एक्टर": "actor",
    "डायरेक्टर": "director", "प्रोड्यूसर": "producer",
    "म्यूजिक": "music", "सॉन्ग": "song", "वीडियो": "video",
    "चैनल": "channel", "शो": "show", "ड्रामा": "drama",
    "कॉमेडी": "comedy", "सीरीज": "series", "सीज़न": "season",
    "एपिसोड": "episode", "यूट्यूब": "youtube",
    "इंस्टाग्राम": "instagram", "फेसबुक": "facebook",
    "ट्विटर": "twitter",

    #  Government / Legal 
    "पॉलिसी": "policy", "गवर्नमेंट": "government",
    "पार्लियामेंट": "parliament", "इलेक्शन": "election",
    "वोट": "vote", "पार्टी": "party", "लीडर": "leader",
    "कमिटी": "committee", "मिनिस्टर": "minister",
    "कमिश्नर": "commissioner", "कोर्ट": "court",
    "जज": "judge", "पुलिस": "police",
    "लाइसेंस": "license", "रजिस्ट्रेशन": "registration",

    #  Clothing / Fashion 
    "फैशन": "fashion", "ड्रेस": "dress", "शर्ट": "shirt",
    "पैंट": "pant", "जींस": "jeans", "जैकेट": "jacket",
    "शूज": "shoes", "बैग": "bag", "ब्रांड": "brand",
    "डिज़ाइन": "design", "डिज़ाइनर": "designer",
    "स्टाइल": "style", "कलर": "color",

    #  General / Common 
    "टाइम": "time", "प्रॉब्लम": "problem", "सॉल्यूशन": "solution",
    "ऑप्शन": "option", "प्लान": "plan", "लिस्ट": "list",
    "ग्रुप": "group", "मेंबर": "member", "फॉर्म": "form",
    "बुक": "book", "पेपर": "paper", "कॉपी": "copy",
    "नंबर": "number", "परसेंट": "percent", "रेट": "rate",
    "लेवल": "level", "टॉप": "top", "बेस्ट": "best",
    "फर्स्ट": "first", "लास्ट": "last", "नेक्स्ट": "next",
    "फ्री": "free", "स्पेशल": "special", "एक्स्ट्रा": "extra",
    "ओके": "ok", "थैंक्स": "thanks", "सॉरी": "sorry",
    "प्लीज": "please", "हैलो": "hello", "बाय": "bye",
    "एड्रेस": "address", "एरिया": "area",
    "पॉइंट": "point", "पर्सनल": "personal",
    "सीक्रेट": "secret", "सेफ": "safe", "रिस्क": "risk",
    "चांस": "chance", "सक्सेस": "success",
    "फेल": "fail", "पास": "pass",
    "फैक्ट": "fact", "ट्रेंड": "trend",
    "इश्यू": "issue", "टॉपिक": "topic",
    "इंपॉर्टेंट": "important", "सीरियस": "serious",
    "नॉर्मल": "normal", "पॉजिटिव": "positive",
    "नेगेटिव": "negative", "रियल": "real",
}

# Reverse lookup: English → Devanagari (for reference)
ENGLISH_TO_DEVANAGARI = {v: k for k, v in ENGLISH_LOANWORDS.items()}

#  Whitelist: Fully absorbed loanwords that are now part of Hindi 
# These should NOT be tagged as [EN] because native Hindi speakers
# consider them Hindi words, not code-switches.
ABSORBED_HINDI_WORDS = {
    # Health / Science — deeply integrated into Hindi vocabulary
    "डॉक्टर", "नर्स", "हॉस्पिटल", "मेडिसिन", "ऑपरेशन",
    "इंजेक्शन", "वैक्सीन", "सर्जरी", "टेस्ट",
    # Transport — no Hindi equivalents commonly used
    "बस", "ट्रेन", "टैक्सी", "ट्रैफिक", "पेट्रोल", "डीज़ल",
    "स्टेशन", "टिकट", "ड्राइवर",
    # Education — used since colonial era
    "स्कूल", "कॉलेज", "यूनिवर्सिटी", "क्लास", "टीचर",
    "प्रोफेसर", "एग्जाम", "रिजल्ट", "डिग्री",
    # Technology — no pure Hindi alternatives exist
    "फोन", "मोबाइल", "कंप्यूटर", "इंटरनेट", "टीवी",
    "रेडियो", "कैमरा",
    # Government / Law — standard administrative vocabulary
    "पुलिस", "कोर्ट", "जज", "वोट", "पार्टी",
    # Food — common everyday words
    "चॉकलेट", "बिस्किट", "केक", "कॉफी", "आइसक्रीम",
    # Sports — standard Hindi usage
    "क्रिकेट", "फुटबॉल", "हॉकी", "मैच", "टीम", "गोल",
    # Clothing — everyday words
    "शर्ट", "पैंट", "जींस",
    # General — deeply naturalized
    "नंबर", "पेपर", "कॉपी", "फिल्म", "बैग",
    "होटल", "रेस्टोरेंट", "लाइसेंस",
}


def tag_english_words(text: str) -> str:
    """
    Tag English words in Hindi text.

    Rules:
      1. Roman script words → always tag as [EN]...[/EN] (clear code-switch)
      2. Devanagari loanwords in dictionary BUT NOT in absorbed whitelist → tag
      3. Absorbed/naturalized words (डॉक्टर, मोबाइल, etc.) → skip
    """
    words = text.split()
    tagged = []

    for word in words:
        # Strip punctuation for lookup, preserve for output
        clean = re.sub(r"[।,!?.\-:;\"'()]+", "", word)

        # Rule 1: Roman script (definitely English code-switch)
        if re.match(r"^[a-zA-Z]+$", clean) and len(clean) >= 2:
            tagged.append(word.replace(clean, f"[EN]{clean}[/EN]"))

        # Rule 2: Devanagari loanword — but NOT if it's absorbed into Hindi
        elif clean in ENGLISH_LOANWORDS and clean not in ABSORBED_HINDI_WORDS:
            tagged.append(word.replace(clean, f"[EN]{clean}[/EN]"))

        else:
            tagged.append(word)

    return " ".join(tagged)


#  Combined Pipeline

def full_pipeline(text: str) -> dict:
    """Run both Part A and Part B on input text."""
    step1 = convert_hindi_numbers(text)
    step2 = tag_english_words(step1)
    return {
        "original": text,
        "after_number_conversion": step1,
        "after_english_tagging": step2,
    }


#  Demo & Verification

def run_demo():
    """Run demo examples for both parts."""

    #  Part A: Number Conversion 
    print("\n" + "=" * 80)
    print("  PART A — Hindi Number Word → Digit Converter")
    
    number_examples = [
        "उसने तीन सौ चौवन रुपये दिये",
        "हमारे गाँव में पच्चीस लोग रहते हैं",
        "एक लाख बीस हज़ार लोगों ने वोट किया",
        "मैंने सत्रह किताबें पढ़ीं",
        "दो हज़ार चौबीस में चुनाव होगा",
    ]

    print(f"\n  {'#':>2}  {'Before':<45} → {'After'}")
    print(f"  {'─'*2}  {'─'*45}   {'─'*40}")
    for i, ex in enumerate(number_examples, 1):
        result = convert_hindi_numbers(ex)
        print(f"  {i:>2}  {ex:<45} → {result}")

    # Edge cases
    print(f"\n  Edge Cases (SHOULD NOT convert):")
    print(f"  {'─'*60}")
    edge_cases = [
        ("दो-चार बातें करनी हैं", "Idiomatic: दो-चार = a few"),
        ("एक-दो बार और कोशिश करो", "Idiomatic: एक-दो = one-two / a couple"),
        ("दोनों भाई स्कूल गए", "दोनों is a pronoun, not a number"),
    ]
    for text, reason in edge_cases:
        result = convert_hindi_numbers(text)
        changed = "✗ KEPT" if text == result else "✓ CONVERTED"
        print(f"    {changed}  {text:<40} → {result}")
        print(f"           Reason: {reason}")

    #  Part B: English Tagging 
    print("\n" + "=" * 80)
    print("  PART B — English Loanword Tagger")
    print(f"  Dictionary size: {len(ENGLISH_LOANWORDS)} loanwords\n")

    english_examples = [
        "मेरा इंटरव्यू बहुत अच्छा गया",
        "कंप्यूटर पर ऑनलाइन क्लास चल रही है",
        "डॉक्टर ने मेडिसिन दी और टेस्ट करवाया",
        "इस प्रोजेक्ट की रिपोर्ट कल submit करनी है",
        "मुझे एक जॉब interview के लिए जाना है",
    ]

    for i, ex in enumerate(english_examples, 1):
        result = tag_english_words(ex)
        print(f"  Example {i}:")
        print(f"    Input : {ex}")
        print(f"    Output: {result}")
        print()

    #  Full Pipeline 
        print("  FULL PIPELINE (Number + English Tagging)")
    
    pipeline_examples = [
        "तीन सौ पचास स्टूडेंट ने ऑनलाइन एग्जाम दिया",
        "एक लाख डॉक्टर ने वैक्सीन लगाई",
        "पच्चीस परसेंट लोग मोबाइल पर इंटरनेट इस्तेमाल करते हैं",
    ]

    for i, ex in enumerate(pipeline_examples, 1):
        out = full_pipeline(ex)
        print(f"\n  Example {i}:")
        print(f"    Original          : {out['original']}")
        print(f"    Numbers converted : {out['after_number_conversion']}")
        print(f"    English tagged    : {out['after_english_tagging']}")


#  CLI

def main():
    parser = argparse.ArgumentParser(description="Hindi ASR Post-Processing Pipeline")
    parser.add_argument("--input", type=str, default=None,
                        help="Single text string to process")
    parser.add_argument("--csv", type=str, default=None,
                        help="CSV file with 'text' or 'raw_asr_output' column")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path")
    parser.add_argument("--demo", action="store_true", default=True,
                        help="Run demo examples (default)")
    args = parser.parse_args()

    print("  HINDI ASR POST-PROCESSING PIPELINE")
    
    if args.input:
        out = full_pipeline(args.input)
        print(f"\n  Original          : {out['original']}")
        print(f"  Numbers converted : {out['after_number_conversion']}")
        print(f"  English tagged    : {out['after_english_tagging']}")

    elif args.csv:
        df = pd.read_csv(args.csv)
        text_col = None
        for c in ["text", "raw_asr_output", "prediction", "hypothesis"]:
            if c in df.columns:
                text_col = c
                break
        if not text_col:
            print(f"  ✗ No text column found. Available: {list(df.columns)}")
            return

        print(f"  Processing {len(df)} rows from column '{text_col}' …")
        df["numbers_converted"] = df[text_col].astype(str).apply(convert_hindi_numbers)
        df["english_tagged"] = df["numbers_converted"].apply(tag_english_words)
        df["cleaned_text"] = df["numbers_converted"]  # final clean version

        out_path = args.output or str(BASE_DIR / "post_processed_output.csv")
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  ✓ Saved to: {out_path}")

    else:
        run_demo()

    print()


if __name__ == "__main__":
    main()
