import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


THEMES = {
    "Studies and exams": [
        "exam", "exams", "study", "studies", "studying",
        "revision", "assignment", "assignments", "college",
        "school", "lecture", "lectures", "homework",
    ],
    "Work and deadlines": [
        "work", "job", "office", "deadline", "deadlines",
        "manager", "colleague", "colleagues", "internship",
        "team meeting", "team meetings",
        "work meeting", "work meetings",
        "office meeting", "office meetings",
        "client meeting", "client meetings",
        "staff meeting", "staff meetings",
        "project deadline", "project deadlines",
    ],
    "Friends and relationships": [
        "friend", "friends", "friendship", "partner",
        "relationship", "relationships", "boyfriend",
        "girlfriend", "breakup", "date",
    ],
    "Family": [
        "family", "mother", "father", "mom", "mum", "dad",
        "parents", "brother", "sister", "siblings",
        "grandmother", "grandfather",
    ],
    "Achievement": [
        "achievement", "achievements", "achieved",
        "accomplished", "success", "succeeded", "proud",
        "passed", "won", "milestone",
    ],
    "Rest and sleep": [
        "sleep", "sleeping", "slept", "insomnia", "rest",
        "resting", "tired", "exhausted", "bedtime", "nap",
    ],
    "Hobbies": [
        "hobby", "hobbies", "painting", "drawing", "music",
        "guitar", "gaming", "gardening", "photography",
        "dancing", "singing", "reading",
    ],
    "Money": [
        "money", "salary", "rent", "bills", "debt",
        "savings", "budget", "expenses", "financial",
        "afford", "loan",
    ],
    "Future plans": [
        "future", "career", "goals", "ambition",
        "aspirations", "next year", "future plans",
        "long term", "long-term",
    ],
}

# Only used for keyword extraction, never for model preprocessing.
KEYWORD_STOP_WORDS = (
    set(ENGLISH_STOP_WORDS)
    | {
        "feel", "feels", "feeling", "felt",
        "really", "just", "quite", "im", "ive",
        "today", "yesterday", "tomorrow",
        "don", "didn", "isn", "wasn", "ve", "ll",
    }
) - {"no", "not", "never", "nor"}

TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)


def extract_keywords(text, top_n=8):
    if top_n <= 0:
        return []

    tokens = TOKEN_PATTERN.findall(
        text.lower().replace("’", "'")
    )

    useful_tokens = [
        token
        for token in tokens
        if len(token) > 2 and token not in KEYWORD_STOP_WORDS
    ]

    # Frequency ties follow first appearance in the text.
    return [
        {"word": word, "count": count}
        for word, count in Counter(useful_tokens).most_common(top_n)
    ]


def detect_themes(text):
    normalized = " ".join(
        text.lower().replace("’", "'").split()
    )

    matches = []

    for theme, phrases in THEMES.items():
        matched_phrases = []

        for phrase in phrases:
            # Word boundaries prevent "exam" matching "example".
            pattern = r"\b" + re.escape(phrase) + r"\b"

            if re.search(pattern, normalized):
                matched_phrases.append(phrase)

        if matched_phrases:
            matches.append({
                "theme": theme,
                "matched_phrases": matched_phrases,
            })

    return matches


def analyze_themes(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter some text.")

    return {
        "themes": detect_themes(text),
        "keywords": extract_keywords(text),
        "method": "Rule-based themes and word-frequency keywords",
    }