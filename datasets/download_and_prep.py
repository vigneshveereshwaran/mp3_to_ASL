"""
HearLink ASL — Dataset Download & Preparation Pipeline
Fetches ASLG-PC12 and How2Sign datasets, parses, cleans, and splits them.
"""

import json
import os
import re
import sys
import hashlib
import zipfile
import tarfile
from pathlib import Path
from typing import Optional

import requests
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from text_normalizer import normalize_english, normalize_gloss, clean_pair

# ──────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent / "data"
ASLG_DIR = DATA_DIR / "aslg_pc12"
HOW2SIGN_DIR = DATA_DIR / "how2sign"

ASLG_PC12_URL = "https://raw.githubusercontent.com/achrafothman/aslg-pc12/master"
ASLG_FILE_COUNT = 12  # corpus_0001 through corpus_0012

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SEED = 42


def ensure_dirs():
    """Create data directories if they don't exist."""
    for d in [DATA_DIR, ASLG_DIR, HOW2SIGN_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress bar. Returns True on success."""
    if dest.exists():
        print(f"  [skip] Already downloaded: {dest.name}")
        return True

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))

        with open(dest, 'wb') as f:
            with tqdm(total=total, unit='B', unit_scale=True, desc=desc or dest.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  [error] Failed to download {url}: {e}")
        if dest.exists():
            dest.unlink()
        return False


# ──────────────────────────────────────────────────
# ASLG-PC12 Dataset
# ──────────────────────────────────────────────────
def download_aslg_pc12():
    """
    Download ASLG-PC12 parallel corpus files.
    The dataset consists of paired .en (English) and .asl (ASL Gloss) files.
    """
    print("\n" + "=" * 60)
    print("Downloading ASLG-PC12 Corpus")
    print("=" * 60)

    ensure_dirs()
    downloaded = 0

    for i in range(1, ASLG_FILE_COUNT + 1):
        file_num = f"{i:04d}"
        for ext in [".en", ".asl"]:
            filename = f"corpus_{file_num}.clean{ext}"
            url = f"{ASLG_PC12_URL}/{filename}"
            dest = ASLG_DIR / filename

            if download_file(url, dest, f"ASLG {file_num}{ext}"):
                downloaded += 1

    print(f"\nDownloaded {downloaded} files to {ASLG_DIR}")
    return downloaded > 0


def parse_aslg_pc12() -> list[tuple[str, str]]:
    """
    Parse ASLG-PC12 corpus files into (English, ASL Gloss) pairs.
    Falls back to generating synthetic data if download fails.
    """
    print("\nParsing ASLG-PC12 corpus...")
    pairs = []

    for i in range(1, ASLG_FILE_COUNT + 1):
        file_num = f"{i:04d}"
        en_file = ASLG_DIR / f"corpus_{file_num}.clean.en"
        asl_file = ASLG_DIR / f"corpus_{file_num}.clean.asl"

        if en_file.exists() and asl_file.exists():
            en_lines = en_file.read_text(encoding='utf-8', errors='ignore').strip().split('\n')
            asl_lines = asl_file.read_text(encoding='utf-8', errors='ignore').strip().split('\n')

            count = min(len(en_lines), len(asl_lines))
            for j in range(count):
                en = en_lines[j].strip()
                asl = asl_lines[j].strip()
                if en and asl:
                    pairs.append((en, asl))

    if not pairs:
        print("  [info] No downloaded files found. Generating synthetic training corpus...")
        pairs = generate_synthetic_aslg_corpus()

    print(f"  Parsed {len(pairs)} sentence pairs from ASLG-PC12")
    return pairs


def generate_synthetic_aslg_corpus() -> list[tuple[str, str]]:
    """
    Generate a synthetic English → ASL Gloss parallel corpus for training.
    Uses rule-based transformations following ASL grammar patterns.
    This is used as fallback if the real dataset cannot be downloaded.
    """
    # Base sentence templates with their ASL gloss equivalents
    templates = [
        # Simple SVO → OSV/Topic-Comment
        ("I like coffee", "COFFEE IX-1 LIKE"),
        ("I love my family", "FAMILY POSS-1 IX-1 LOVE"),
        ("She reads books", "BOOK IX-3 READ"),
        ("He drives a car", "CAR IX-3 DRIVE"),
        ("We eat dinner", "DINNER IX-1-PLURAL EAT"),
        ("They play soccer", "SOCCER IX-3-PLURAL PLAY"),
        ("The dog runs fast", "DOG RUN FAST"),
        ("The cat sleeps on the bed", "BED CAT SLEEP"),
        ("My mother cooks food", "FOOD MOTHER POSS-1 COOK"),
        ("The teacher explains the lesson", "LESSON TEACHER EXPLAIN"),

        # Questions (WH-questions typically go at end in ASL)
        ("What is your name?", "NAME YOU WHAT"),
        ("Where do you live?", "LIVE YOU WHERE"),
        ("How old are you?", "AGE YOU HOW-OLD"),
        ("When is the meeting?", "MEETING WHEN"),
        ("Who is your teacher?", "TEACHER POSS-2 WHO"),
        ("What time is it?", "TIME WHAT"),
        ("Why are you sad?", "SAD YOU WHY"),
        ("How are you?", "YOU HOW"),
        ("What do you want?", "WANT YOU WHAT"),
        ("Where is the bathroom?", "BATHROOM WHERE"),

        # Yes/No questions (raised eyebrows marker)
        ("Do you understand?", "UNDERSTAND YOU"),
        ("Are you hungry?", "HUNGRY YOU"),
        ("Is she coming?", "COME IX-3"),
        ("Did you finish?", "FINISH YOU"),
        ("Can you help me?", "HELP IX-1 YOU CAN"),

        # Negation
        ("I don't understand", "UNDERSTAND IX-1 NOT"),
        ("She can't come", "COME IX-3 CAN NOT"),
        ("He doesn't like vegetables", "VEGETABLE IX-3 LIKE NOT"),
        ("They won't go", "GO IX-3-PLURAL WILL NOT"),
        ("I have never been there", "THERE IX-1 GO FINISH NEVER"),

        # Time-Topic-Comment structure
        ("Yesterday I went to the store", "YESTERDAY STORE IX-1 GO FINISH"),
        ("Tomorrow we will have a test", "TOMORROW TEST IX-1-PLURAL HAVE WILL"),
        ("Last week she visited her grandmother", "LAST-WEEK GRANDMOTHER POSS-3 IX-3 VISIT FINISH"),
        ("I will call you later", "LATER IX-2 IX-1 CALL WILL"),
        ("She already finished her homework", "ALREADY HOMEWORK POSS-3 IX-3 FINISH"),
        ("Next month I start a new job", "NEXT-MONTH JOB NEW IX-1 START"),
        ("Before I go I need to eat", "BEFORE GO IX-1 EAT NEED"),
        ("After school we play basketball", "SCHOOL AFTER BASKETBALL IX-1-PLURAL PLAY"),

        # Descriptions / Adjectives (follow the noun in ASL)
        ("The tall man walks slowly", "MAN TALL WALK SLOW"),
        ("I have a big red car", "CAR RED BIG IX-1 HAVE"),
        ("The beautiful flowers bloom in spring", "SPRING FLOWER BEAUTIFUL BLOOM"),
        ("She wears a blue dress", "DRESS BLUE IX-3 WEAR"),
        ("The small child laughs loudly", "CHILD SMALL LAUGH LOUD"),

        # Commands / Imperatives
        ("Please sit down", "SIT PLEASE"),
        ("Stop talking", "TALK STOP"),
        ("Come here", "HERE COME"),
        ("Look at me", "IX-1 LOOK"),
        ("Wait a moment", "MOMENT WAIT"),
        ("Open the door", "DOOR OPEN"),
        ("Close the window", "WINDOW CLOSE"),
        ("Turn off the light", "LIGHT OFF TURN"),

        # Conditional
        ("If it rains I will stay home", "RAIN IF HOME IX-1 STAY WILL"),
        ("If you study hard you will pass", "STUDY HARD YOU IF PASS YOU WILL"),

        # Emotions / States
        ("I am happy", "HAPPY IX-1"),
        ("She is tired", "TIRED IX-3"),
        ("He feels sick", "SICK IX-3 FEEL"),
        ("We are excited about the trip", "TRIP IX-1-PLURAL EXCITED"),
        ("They are angry", "ANGRY IX-3-PLURAL"),
        ("I am confused", "CONFUSED IX-1"),
        ("She is surprised", "SURPRISED IX-3"),
        ("He is bored", "BORED IX-3"),
        ("I am worried about the exam", "EXAM IX-1 WORRIED"),
        ("She is proud of her son", "SON POSS-3 IX-3 PROUD"),

        # Numbers / Quantities
        ("I have three cats", "CAT THREE IX-1 HAVE"),
        ("She bought two tickets", "TICKET TWO IX-3 BUY FINISH"),
        ("There are many people here", "HERE PEOPLE MANY"),
        ("He ate five apples", "APPLE FIVE IX-3 EAT FINISH"),
        ("We need more time", "TIME MORE IX-1-PLURAL NEED"),

        # Locations / Spatial
        ("The book is on the table", "TABLE BOOK ON"),
        ("The cat is under the chair", "CHAIR CAT UNDER"),
        ("She lives in New York", "FS-NEW-YORK IX-3 LIVE"),
        ("The hospital is near the school", "SCHOOL HOSPITAL NEAR"),
        ("He works at the office", "OFFICE IX-3 WORK"),

        # Daily activities
        ("I wake up early every morning", "EVERY-MORNING EARLY IX-1 WAKE-UP"),
        ("She takes a shower before breakfast", "BREAKFAST BEFORE SHOWER IX-3 TAKE"),
        ("He brushes his teeth twice a day", "DAY TWICE TEETH POSS-3 IX-3 BRUSH"),
        ("We go to church on Sundays", "SUNDAY CHURCH IX-1-PLURAL GO"),
        ("They watch television at night", "NIGHT TELEVISION IX-3-PLURAL WATCH"),

        # Communication
        ("I want to learn sign language", "SIGN LANGUAGE LEARN IX-1 WANT"),
        ("She speaks three languages", "LANGUAGE THREE IX-3 SPEAK"),
        ("He told me a story", "STORY IX-1 IX-3 TELL"),
        ("We had a long conversation", "CONVERSATION LONG IX-1-PLURAL HAVE FINISH"),
        ("Please repeat that", "THAT REPEAT PLEASE"),
        ("I don't know the answer", "ANSWER IX-1 KNOW NOT"),
        ("Can you spell your name?", "NAME POSS-2 SPELL YOU CAN"),
        ("She explained the problem clearly", "PROBLEM IX-3 EXPLAIN CLEAR"),

        # Family
        ("My brother is a doctor", "BROTHER POSS-1 DOCTOR"),
        ("Her sister lives far away", "SISTER POSS-3 FAR LIVE"),
        ("His parents are both teachers", "PARENT POSS-3 BOTH TEACHER"),
        ("Our grandparents visited us last summer", "LAST-SUMMER GRANDPARENT POSS-1-PLURAL IX-1-PLURAL VISIT FINISH"),
        ("The baby is sleeping", "BABY SLEEP"),

        # Weather
        ("It is raining outside", "OUTSIDE RAIN"),
        ("The weather is cold today", "TODAY WEATHER COLD"),
        ("It will snow tomorrow", "TOMORROW SNOW WILL"),
        ("The sun is shining brightly", "SUN SHINE BRIGHT"),
        ("There is a storm coming", "STORM COME"),

        # Food
        ("I am hungry", "HUNGRY IX-1"),
        ("What do you want to eat?", "EAT WANT YOU WHAT"),
        ("She ordered a pizza", "PIZZA IX-3 ORDER FINISH"),
        ("The food is delicious", "FOOD DELICIOUS"),
        ("He is allergic to peanuts", "PEANUT IX-3 ALLERGIC"),
        ("We cooked dinner together", "TOGETHER DINNER IX-1-PLURAL COOK FINISH"),
        ("I prefer tea over coffee", "TEA COFFEE IX-1 PREFER TEA"),

        # School / Education
        ("I study at the university", "UNIVERSITY IX-1 STUDY"),
        ("She passed her final exam", "EXAM FINAL POSS-3 IX-3 PASS FINISH"),
        ("The homework is due tomorrow", "TOMORROW HOMEWORK DUE"),
        ("He graduated last year", "LAST-YEAR IX-3 GRADUATE FINISH"),
        ("We have a class at nine", "NINE CLASS IX-1-PLURAL HAVE"),

        # Work
        ("I need to finish this project", "PROJECT THIS IX-1 FINISH NEED"),
        ("She got a promotion", "PROMOTION IX-3 GET FINISH"),
        ("He is looking for a job", "JOB IX-3 LOOK-FOR"),
        ("We have a meeting at two", "TWO MEETING IX-1-PLURAL HAVE"),
        ("The boss wants to see you", "BOSS IX-2 SEE WANT"),

        # Health
        ("I have a headache", "HEADACHE IX-1 HAVE"),
        ("She needs to see a doctor", "DOCTOR IX-3 SEE NEED"),
        ("He broke his arm", "ARM POSS-3 IX-3 BREAK FINISH"),
        ("Take this medicine twice a day", "MEDICINE THIS DAY TWICE TAKE"),
        ("I feel better now", "NOW BETTER IX-1 FEEL"),

        # Transportation
        ("I take the bus to work", "WORK BUS IX-1 TAKE"),
        ("She missed the train", "TRAIN IX-3 MISS FINISH"),
        ("He rides his bike to school", "SCHOOL BIKE POSS-3 IX-3 RIDE"),
        ("The flight is delayed", "FLIGHT DELAY"),
        ("We drove for three hours", "HOUR THREE IX-1-PLURAL DRIVE FINISH"),

        # Shopping
        ("How much does this cost?", "THIS COST HOW-MUCH"),
        ("I want to buy a new phone", "PHONE NEW IX-1 BUY WANT"),
        ("She returned the dress", "DRESS IX-3 RETURN FINISH"),
        ("The store closes at nine", "NINE STORE CLOSE"),
        ("We need to buy groceries", "GROCERY IX-1-PLURAL BUY NEED"),

        # Technology
        ("I forgot my password", "PASSWORD POSS-1 IX-1 FORGET"),
        ("The computer is broken", "COMPUTER BREAK"),
        ("She sent me an email", "EMAIL IX-1 IX-3 SEND FINISH"),
        ("He is downloading the app", "APP IX-3 DOWNLOAD"),
        ("The internet is slow", "INTERNET SLOW"),

        # Hobbies
        ("I enjoy reading books", "BOOK READ IX-1 ENJOY"),
        ("She paints beautiful pictures", "PICTURE BEAUTIFUL IX-3 PAINT"),
        ("He plays guitar every evening", "EVERY-EVENING GUITAR IX-3 PLAY"),
        ("We go hiking on weekends", "WEEKEND HIKE IX-1-PLURAL GO"),
        ("They love watching movies", "MOVIE WATCH IX-3-PLURAL LOVE"),

        # Greetings / Social
        ("Hello, how are you?", "HELLO YOU HOW"),
        ("Nice to meet you", "MEET YOU NICE"),
        ("Goodbye, see you later", "LATER SEE-YOU BYE"),
        ("Thank you very much", "THANK-YOU VERY-MUCH"),
        ("I'm sorry for being late", "LATE IX-1 SORRY"),
        ("Excuse me, can I ask a question?", "EXCUSE-ME QUESTION IX-1 ASK CAN"),
        ("Happy birthday!", "BIRTHDAY HAPPY"),
        ("Congratulations on your graduation!", "GRADUATE POSS-2 CONGRATULATIONS"),
        ("Good morning", "MORNING GOOD"),
        ("Good night, sleep well", "NIGHT GOOD SLEEP WELL"),
    ]

    # Generate variations by substituting nouns/verbs
    nouns = ["BOOK", "CAR", "HOUSE", "PHONE", "DOG", "CAT", "FOOD", "WATER",
             "SCHOOL", "WORK", "FRIEND", "FAMILY", "MONEY", "TIME", "MUSIC",
             "MOVIE", "GAME", "COMPUTER", "DOOR", "WINDOW", "TABLE", "CHAIR",
             "TEACHER", "STUDENT", "DOCTOR", "CHILD", "BABY", "MOTHER", "FATHER"]

    verbs = ["LIKE", "WANT", "NEED", "HAVE", "SEE", "HEAR", "KNOW", "THINK",
             "FEEL", "LOVE", "HATE", "HELP", "GIVE", "TAKE", "MAKE", "GO",
             "COME", "EAT", "DRINK", "SLEEP", "WORK", "PLAY", "READ", "WRITE",
             "BUY", "SELL", "FIND", "LOSE", "START", "FINISH"]

    adjectives = ["BIG", "SMALL", "GOOD", "BAD", "NEW", "OLD", "HAPPY", "SAD",
                  "FAST", "SLOW", "HOT", "COLD", "EASY", "HARD", "BEAUTIFUL",
                  "IMPORTANT", "INTERESTING", "BORING", "EXPENSIVE", "CHEAP"]

    # Generate additional pairs from simple patterns
    additional = []
    en_nouns = [n.lower() for n in nouns]
    en_verbs_map = {
        "LIKE": "like", "WANT": "want", "NEED": "need", "HAVE": "have",
        "SEE": "see", "HEAR": "hear", "KNOW": "know", "THINK": "think about",
        "FEEL": "feel", "LOVE": "love", "HATE": "hate", "HELP": "help with",
        "BUY": "buy", "FIND": "find", "MAKE": "make",
    }
    en_adj_map = {a: a.lower() for a in adjectives}

    for noun in nouns[:15]:
        en_n = noun.lower()
        for verb_asl, verb_en in list(en_verbs_map.items())[:10]:
            # "I [verb] [noun]" → "[NOUN] IX-1 [VERB]"
            additional.append((f"i {verb_en} the {en_n}", f"{noun} IX-1 {verb_asl}"))
            # "She [verb] [noun]" → "[NOUN] IX-3 [VERB]"
            additional.append((f"she {verb_en}s the {en_n}", f"{noun} IX-3 {verb_asl}"))

        for adj_asl, adj_en in list(en_adj_map.items())[:10]:
            # "The [noun] is [adj]" → "[NOUN] [ADJ]"
            additional.append((f"the {en_n} is {adj_en}", f"{noun} {adj_asl}"))

    # Combine templates + generated
    all_pairs = templates + additional

    print(f"  Generated {len(all_pairs)} synthetic training pairs")
    return all_pairs


# ──────────────────────────────────────────────────
# How2Sign Dataset
# ──────────────────────────────────────────────────
def parse_how2sign_transcripts() -> list[tuple[str, str]]:
    """
    Parse How2Sign transcript files if available.
    Looks for TSV/CSV annotation files with English and gloss columns.
    """
    print("\nParsing How2Sign transcripts...")
    pairs = []

    # Look for annotation files
    annotation_patterns = [
        HOW2SIGN_DIR / "*.tsv",
        HOW2SIGN_DIR / "*.csv",
        HOW2SIGN_DIR / "annotations" / "*.tsv",
    ]

    import glob
    tsv_files = []
    for pattern in annotation_patterns:
        tsv_files.extend(glob.glob(str(pattern)))

    if not tsv_files:
        print("  [info] No How2Sign annotation files found.")
        print("  [info] To add How2Sign data, download from: https://how2sign.github.io/")
        print("  [info] Place TSV annotation files in: datasets/data/how2sign/")
        return pairs

    for tsv_file in tsv_files:
        try:
            import pandas as pd
            df = pd.read_csv(tsv_file, sep='\t', on_bad_lines='skip')

            # Common column names in How2Sign
            en_cols = [c for c in df.columns if any(k in c.lower() for k in ['english', 'sentence', 'text', 'transcript'])]
            gl_cols = [c for c in df.columns if any(k in c.lower() for k in ['gloss', 'sign', 'asl'])]

            if en_cols and gl_cols:
                en_col = en_cols[0]
                gl_col = gl_cols[0]
                for _, row in df.iterrows():
                    en = str(row[en_col]).strip()
                    gl = str(row[gl_col]).strip()
                    if en and gl and en != 'nan' and gl != 'nan':
                        pairs.append((en, gl))
        except Exception as e:
            print(f"  [warning] Error parsing {tsv_file}: {e}")

    print(f"  Parsed {len(pairs)} pairs from How2Sign")
    return pairs


# ──────────────────────────────────────────────────
# Data Splitting & Export
# ──────────────────────────────────────────────────
def split_data(pairs: list[tuple[str, str]]) -> dict[str, list[tuple[str, str]]]:
    """
    Split data into train/val/test with deterministic shuffling.
    """
    rng = np.random.RandomState(SEED)
    indices = np.arange(len(pairs))
    rng.shuffle(indices)

    n_train = int(len(pairs) * TRAIN_RATIO)
    n_val = int(len(pairs) * VAL_RATIO)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return {
        "train": [pairs[i] for i in train_idx],
        "val": [pairs[i] for i in val_idx],
        "test": [pairs[i] for i in test_idx],
    }


def save_jsonl(pairs: list[tuple[str, str]], path: Path):
    """Save pairs as JSONL format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for en, gl in pairs:
            f.write(json.dumps({"en": en, "gloss": gl}, ensure_ascii=False) + '\n')
    print(f"  Saved {len(pairs)} pairs to {path}")


def main():
    """Main data preparation pipeline."""
    print("=" * 60)
    print("HearLink ASL — Dataset Preparation Pipeline")
    print("=" * 60)

    ensure_dirs()

    # Step 1: Download ASLG-PC12
    download_aslg_pc12()

    # Step 2: Parse all datasets
    aslg_pairs = parse_aslg_pc12()
    how2sign_pairs = parse_how2sign_transcripts()

    # Step 3: Combine and clean
    all_pairs = aslg_pairs + how2sign_pairs
    print(f"\nTotal raw pairs: {len(all_pairs)}")

    # Clean and normalize
    from text_normalizer import normalize_batch
    cleaned_pairs = normalize_batch(all_pairs)
    print(f"After cleaning: {len(cleaned_pairs)}")

    # Step 4: Split
    splits = split_data(cleaned_pairs)
    print(f"\nSplit sizes:")
    for name, data in splits.items():
        print(f"  {name}: {len(data)} pairs")

    # Step 5: Save
    print("\nSaving splits...")
    for name, data in splits.items():
        save_jsonl(data, DATA_DIR / f"{name}.jsonl")

    # Step 6: Stats
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)

    all_en_words = set()
    all_gl_tokens = set()
    for en, gl in cleaned_pairs:
        all_en_words.update(en.split())
        all_gl_tokens.update(gl.split())

    print(f"Total sentence pairs: {len(cleaned_pairs)}")
    print(f"Unique English words: {len(all_en_words)}")
    print(f"Unique Gloss tokens:  {len(all_gl_tokens)}")

    en_lengths = [len(en.split()) for en, _ in cleaned_pairs]
    gl_lengths = [len(gl.split()) for _, gl in cleaned_pairs]
    print(f"Avg English length:   {np.mean(en_lengths):.1f} words")
    print(f"Avg Gloss length:     {np.mean(gl_lengths):.1f} tokens")
    print(f"Max English length:   {max(en_lengths)} words")
    print(f"Max Gloss length:     {max(gl_lengths)} tokens")

    print("\nSample pairs:")
    for en, gl in cleaned_pairs[:5]:
        print(f"  EN:    {en}")
        print(f"  GLOSS: {gl}")
        print()

    print("[SUCCESS] Dataset preparation complete!")
    return splits


if __name__ == "__main__":
    main()
