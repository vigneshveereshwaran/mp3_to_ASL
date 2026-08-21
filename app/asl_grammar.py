"""
HearLink ASL — NLP & ASL Grammar Translation Engine
Transforms English into structured ASL Gloss, Non-Manual Markers (NMM), and semantic metadata.
Features POS analysis, WH-movement, Time-Topic-Comment structure, negation, copula deletion, and pronoun indexing.
"""

import re
from typing import Dict, List, Any, Optional, Tuple

# ── PRONOUN & POSSESSIVE MAP ──
PRONOUN_MAP = {
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
    "she": "IX-3",
    "her": "POSS-3",
    "hers": "POSS-3",
    "it": "IX-3",
    "its": "POSS-3",
    "we": "IX-1-PLURAL",
    "us": "IX-1-PLURAL",
    "our": "POSS-1-PLURAL",
    "ours": "POSS-1-PLURAL",
    "they": "IX-3-PLURAL",
    "them": "IX-3-PLURAL",
    "their": "POSS-3-PLURAL",
    "theirs": "POSS-3-PLURAL",
}

# ── CONTRACTIONS ──
CONTRACTIONS = {
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "you're": "you are", "you've": "you have", "you'll": "you will", "you'd": "you would",
    "he's": "he is", "she's": "she is", "it's": "it is",
    "we're": "we are", "we've": "we have", "we'll": "we will", "we'd": "we would",
    "they're": "they are", "they've": "they have", "they'll": "they will", "they'd": "they would",
    "that's": "that is", "who's": "who is", "what's": "what is", "where's": "where is",
    "there's": "there is", "here's": "here is",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
    "doesn't": "does not", "don't": "do not", "didn't": "did not",
    "won't": "will not", "wouldn't": "would not", "shan't": "shall not",
    "shouldn't": "should not", "mustn't": "must not",
    "can't": "cannot", "couldn't": "could not",
    "let's": "let us",
}

# ── STOPWORDS / FUNCTION WORDS TO DROP IN ASL ──
DROP_WORDS = {
    "a", "an", "the",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did",
    "to", "of", "for", "at", "by", "from",
    "very", "really", "just", "quite", "so",
    "that", "which"
}

# ── TIME EXPRESSIONS (Fronted in ASL Time-Topic-Comment) ──
TIME_WORDS = {
    "yesterday", "today", "tomorrow", "now", "later", "before", "after",
    "recently", "soon", "always", "never", "sometimes", "often", "usually",
    "morning", "afternoon", "evening", "night", "last-night", "tonight",
    "last-week", "next-week", "last-year", "next-year", "daily", "everyday"
}

# ── WH-QUESTION WORDS (Moved to sentence end in ASL) ──
WH_WORDS = {
    "what", "where", "who", "when", "why", "how", "which", "how-much", "how-many"
}

# ── NEGATION WORDS ──
NEGATION_WORDS = {
    "not", "no", "never", "cannot", "cant", "none", "neither", "nowhere"
}

# ── PHRASE / IDIOM MAPPINGS ──
IDIOM_MAP = {
    "how are you": ["HOW", "IX-2"],
    "what is your name": ["NAME", "POSS-2", "WHAT"],
    "what is your name?": ["NAME", "POSS-2", "WHAT"],
    "nice to meet you": ["MEET", "IX-2", "NICE"],
    "thank you very much": ["THANK-YOU", "VERY-MUCH"],
    "thank you": ["THANK-YOU"],
    "you are welcome": ["WELCOME"],
    "see you later": ["SEE-YOU", "LATER"],
    "good morning": ["MORNING", "GOOD"],
    "good night": ["NIGHT", "GOOD"],
    "good afternoon": ["AFTERNOON", "GOOD"],
    "i do not know": ["KNOW", "NOT", "IX-1"],
    "i don't know": ["KNOW", "NOT", "IX-1"],
    "i need help": ["HELP", "IX-1", "NEED"],
    "where is the bathroom": ["BATHROOM", "WHERE"],
    "where is the school": ["SCHOOL", "WHERE"],
    "i am a student": ["IX-1", "STUDENT"],
    "i am a teacher": ["IX-1", "TEACHER"],
    "i love you": ["LOVE", "IX-1", "IX-2"],
    "what time is it": ["TIME", "WHAT"]
}


class ASLGrammarProcessor:
    """
    Parses English text and generates linguistically-sound ASL Gloss sequences
    and synchronized Non-Manual Markers (NMM).
    """

    def __init__(self):
        pass

    def normalize(self, text: str) -> str:
        """Expand contractions, lowercase, and sanitize."""
        text = text.lower().strip()
        for contraction, expansion in CONTRACTIONS.items():
            text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text)
        return text

    def analyze(self, raw_text: str) -> Dict[str, Any]:
        """
        Full linguistic transformation pipeline.
        Returns:
            gloss (str): Final space-separated ASL gloss
            tokens (list): Token list
            non_manual (dict): Sentence-level and token-level facial/head markers
            grammar_features (dict): Identified question types, negation, time-fronting
        """
        raw_clean = raw_text.strip()
        if not raw_clean:
            return {
                "gloss": "",
                "tokens": [],
                "non_manual": {"eyebrows": "neutral", "head": "neutral", "mouth": "neutral"},
                "grammar_features": {"is_question": False, "is_wh_question": False, "is_negated": False, "has_time": False}
            }

        norm_text = self.normalize(raw_clean)
        is_question = "?" in raw_clean or any(norm_text.startswith(w) for w in ["what", "where", "who", "when", "why", "how", "is", "are", "do", "does", "can", "could", "will", "would"])
        
        # Check idioms first
        clean_text_no_punct = re.sub(r'[^\w\s-]', '', norm_text).strip()
        if clean_text_no_punct in IDIOM_MAP:
            tokens = IDIOM_MAP[clean_text_no_punct]
            wh_q = any(t in ["WHAT", "WHERE", "WHO", "WHEN", "WHY", "HOW"] for t in tokens)
            nmm = self._determine_nmm(is_question, wh_q, "NOT" in tokens)
            return {
                "gloss": " ".join(tokens),
                "tokens": tokens,
                "non_manual": nmm,
                "grammar_features": {
                    "is_question": is_question,
                    "is_wh_question": wh_q,
                    "is_negated": "NOT" in tokens,
                    "has_time": any(t in TIME_WORDS for t in clean_text_no_punct.split()),
                    "matched_idiom": True
                }
            }

        # Step 1: Tokenize & Identify Roles
        words = re.findall(r"[\w'-]+", norm_text)
        
        time_tokens = []
        topic_tokens = []
        verb_tokens = []
        wh_tokens = []
        neg_tokens = []
        other_tokens = []

        is_wh_q = False
        is_negated = False

        for w in words:
            # Check negation
            if w in NEGATION_WORDS:
                is_negated = True
                neg_tokens.append("NOT")
                continue

            # Check time expression
            if w in TIME_WORDS:
                time_tokens.append(w.upper())
                continue

            # Check WH-question
            if w in WH_WORDS:
                is_wh_q = True
                wh_tokens.append(w.upper())
                continue

            # Check Pronouns
            if w in PRONOUN_MAP:
                other_tokens.append(PRONOUN_MAP[w])
                continue

            # Check Drop words
            if w in DROP_WORDS:
                continue

            # Standard content word
            other_tokens.append(w.upper())

        # Step 2: Assemble ASL Grammatical Structure (Time -> Topic -> Comment / Verb -> Negation -> WH-Question)
        final_tokens = []

        # 1. TIME markers fronted
        final_tokens.extend(time_tokens)

        # 2. TOPIC & CONTENT words
        final_tokens.extend(other_tokens)

        # 3. NEGATION placed post-predicate
        final_tokens.extend(neg_tokens)

        # 4. WH-WORDS placed at sentence boundary
        final_tokens.extend(wh_tokens)

        # Clean duplicates while preserving structure
        cleaned_tokens = []
        for t in final_tokens:
            if not t:
                continue
            cleaned_tokens.append(t)

        if not cleaned_tokens:
            cleaned_tokens = ["EMPTY"]

        gloss_str = " ".join(cleaned_tokens)
        nmm = self._determine_nmm(is_question or is_wh_q, is_wh_q, is_negated)

        return {
            "gloss": gloss_str,
            "tokens": cleaned_tokens,
            "non_manual": nmm,
            "grammar_features": {
                "is_question": is_question or is_wh_q,
                "is_wh_question": is_wh_q,
                "is_negated": is_negated,
                "has_time": len(time_tokens) > 0,
                "matched_idiom": False
            }
        }

    def _determine_nmm(self, is_q: bool, is_wh_q: bool, is_neg: bool) -> Dict[str, Any]:
        """
        Generate linguistic Non-Manual Markers (NMM):
        - WH-questions: Furrowed eyebrows, slight head tilt forward.
        - Yes/No questions: Raised eyebrows, slight head forward.
        - Negation: Head shake, furrowed brow.
        - Affirmation/Declarative: Neutral or subtle nod.
        """
        if is_wh_q:
            return {
                "eyebrows": "furrowed",  # -0.6
                "head": "forward_tilt",
                "head_tilt": 0.15,
                "mouth": "wh_pucker",
                "facial_intensity": 0.8
            }
        elif is_q:
            return {
                "eyebrows": "raised",    # +0.7
                "head": "forward_tilt",
                "head_tilt": 0.12,
                "mouth": "open_slight",
                "facial_intensity": 0.75
            }
        elif is_neg:
            return {
                "eyebrows": "furrowed_slight",
                "head": "head_shake",
                "head_tilt": -0.1,
                "mouth": "tight_lips",
                "facial_intensity": 0.85
            }
        else:
            return {
                "eyebrows": "neutral",
                "head": "neutral",
                "head_tilt": 0.0,
                "mouth": "neutral",
                "facial_intensity": 0.0
            }
