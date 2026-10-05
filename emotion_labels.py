# Target labels for the next dataset/model.
# Existing six-class models remain unchanged.

CORE_EMOTIONS = [
    "joy",
    "trust",
    "fear",
    "surprise",
    "sadness",
    "disgust",
    "anger",
    "anticipation",
]

EXTRA_EMOTIONS = [
    "love",
]

EMOTION_LABELS = CORE_EMOTIONS + EXTRA_EMOTIONS

ANNOTATION_STATES = [
    "neutral",  # No clear emotion expressed.
    "unclear",  # Insufficient or ambiguous evidence.
]

# Multiple emotions are stored as multiple labels:
# ["sadness", "joy"], rather than a separate "mixed" emotion.