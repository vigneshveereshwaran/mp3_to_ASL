"""
HearLink ASL — Text Normalizer
Cleans and normalizes English text and ASL Gloss tokens for training.
"""

import re
import string
from typing import Optional


# Common English contractions → expanded forms
CONTRACTIONS = {
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "you're": "you are", "you've": "you have", "you'll": "you will", "you'd": "you would",
    "he's": "he is", "she's": "she is", "it's": "it is",
    "we're": "we are", "we've": "we have", "we'll": "we will", "we'd": "we would",
    "they're": "they are", "they've": "they have", "they'll": "they will", "they'd": "they would",
    "that's": "that is", "who's": "who is", "what's": "what is",
    "there's": "there is", "here's": "here is",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
    "doesn't": "does not", "don't": "do not", "didn't": "did not",
    "won't": "will not", "wouldn't": "would not", "shan't": "shall not",
    "shouldn't": "should not", "mustn't": "must not",
    "can't": "cannot", "couldn't": "could not",
    "let's": "let us",
}

# ASL pronoun index mappings
ASL_PRONOUN_MAP = {
    "i": "IX-1",
    "me": "IX-1",
    "my": "POSS-1",
    "mine": "POSS-1",
    "myself": "SELF-1",
    "you": "IX-2",
    "your": "POSS-2",
    "yours": "POSS-2",
    "yourself": "SELF-2",
    "he": "IX-3",
    "him": "IX-3",
    "his": "POSS-3",
    "himself": "SELF-3",
    "she": "IX-3",
    "her": "IX-3",
    "hers": "POSS-3",
    "herself": "SELF-3",
    "it": "IX-3",
    "its": "POSS-3",
    "itself": "SELF-3",
    "we": "IX-1-PLURAL",
    "us": "IX-1-PLURAL",
    "our": "POSS-1-PLURAL",
    "ours": "POSS-1-PLURAL",
    "they": "IX-3-PLURAL",
    "them": "IX-3-PLURAL",
    "their": "POSS-3-PLURAL",
    "theirs": "POSS-3-PLURAL",
}

# English function words typically dropped in ASL
ASL_DROP_WORDS = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had",
    "to", "of", "for", "with", "at", "by", "from", "in", "on",
    "that", "which", "who", "whom",
    "very", "really", "just", "quite",
}

# Time-related words that should front the ASL sentence
TIME_WORDS = {
    "yesterday", "today", "tomorrow", "now", "later", "before",
    "after", "already", "recently", "soon", "always", "never",
    "sometimes", "often", "usually", "morning", "afternoon",
    "evening", "night", "last", "next", "ago",
}


def normalize_english(text: str) -> str:
    """
    Normalize English input text for model consumption.
    - Lowercase
    - Expand contractions
    - Strip punctuation
    - Collapse whitespace
    """
    text = text.lower().strip()

    # Expand contractions
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text)

    # Remove punctuation (keep hyphens in compound words)
    text = re.sub(r'[^\w\s-]', '', text)

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def normalize_gloss(gloss: str) -> str:
    """
    Normalize ASL Gloss text.
    - Uppercase all tokens
    - Preserve fs- (fingerspelling) prefixes
    - Preserve special markers (IX-, POSS-, CL-)
    - Remove extraneous punctuation
    - Collapse whitespace
    """
    gloss = gloss.strip()

    # Uppercase everything
    gloss = gloss.upper()

    # Clean punctuation but preserve hyphens (used in ASL notation)
    gloss = re.sub(r'[^\w\s\-]', '', gloss)

    # Normalize fingerspelling prefix
    gloss = re.sub(r'\bFS\s*-\s*', 'FS-', gloss)

    # Collapse whitespace
    gloss = re.sub(r'\s+', ' ', gloss).strip()

    return gloss


def english_to_asl_grammar(text: str) -> str:
    """
    Apply basic English → ASL grammar transformations:
    1. Extract time markers and front them (Time-Topic-Comment)
    2. Drop function words (articles, copulas, prepositions)
    3. Map pronouns to ASL index signs
    4. Uppercase the result (gloss convention)

    This is a heuristic baseline — the neural model will learn finer patterns.
    """
    text = normalize_english(text)
    words = text.split()

    # Step 1: Extract time words → move to front
    time_tokens = []
    remaining = []
    for word in words:
        if word in TIME_WORDS:
            time_tokens.append(word.upper())
        else:
            remaining.append(word)

    # Step 2: Process remaining words
    gloss_tokens = []
    for word in remaining:
        # Drop function words
        if word in ASL_DROP_WORDS:
            continue
        # Map pronouns
        if word in ASL_PRONOUN_MAP:
            gloss_tokens.append(ASL_PRONOUN_MAP[word])
        else:
            gloss_tokens.append(word.upper())

    # Combine: TIME + TOPIC/COMMENT
    result = time_tokens + gloss_tokens

    return ' '.join(result) if result else ''


def handle_fingerspelling(word: str) -> str:
    """
    Convert a word to fingerspelling notation.
    Each letter becomes a separate sign: fs-WORD
    Used for proper nouns and words without a direct ASL sign.
    """
    word = word.upper().strip()
    if not word:
        return ''
    return f'FS-{word}'


def detect_fingerspelled(gloss: str) -> list[str]:
    """
    Extract fingerspelled tokens from a gloss string.
    Returns list of words that are fingerspelled (prefixed with FS-).
    """
    tokens = gloss.split()
    return [t for t in tokens if t.startswith('FS-')]


def clean_pair(english: str, gloss: str) -> Optional[tuple[str, str]]:
    """
    Clean and validate an English-Gloss pair.
    Returns None if the pair should be discarded.
    """
    en = normalize_english(english)
    gl = normalize_gloss(gloss)

    # Discard empty pairs
    if not en or not gl:
        return None

    # Discard very short pairs (likely noise)
    if len(en.split()) < 2 or len(gl.split()) < 1:
        return None

    # Discard very long pairs (likely parsing errors)
    if len(en.split()) > 100 or len(gl.split()) > 80:
        return None

    return (en, gl)


def normalize_batch(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Normalize and filter a batch of (English, Gloss) pairs.
    Removes duplicates and invalid entries.
    """
    seen = set()
    results = []

    for en, gl in pairs:
        cleaned = clean_pair(en, gl)
        if cleaned is None:
            continue

        # Deduplicate by English text
        if cleaned[0] in seen:
            continue
        seen.add(cleaned[0])
        results.append(cleaned)

    return results


if __name__ == "__main__":
    # Quick demo
    test_sentences = [
        ("I'm going to the store tomorrow", "TOMORROW IX-1 GO STORE"),
        ("She doesn't like coffee", "IX-3 NOT LIKE COFFEE"),
        ("The cat is sitting on the mat", "CAT SIT MAT"),
    ]

    print("=" * 60)
    print("HearLink ASL — Text Normalizer Demo")
    print("=" * 60)

    for en, gl in test_sentences:
        print(f"\nEnglish:    {en}")
        print(f"Normalized: {normalize_english(en)}")
        print(f"Heuristic:  {english_to_asl_grammar(en)}")
        print(f"Gloss:      {normalize_gloss(gl)}")
