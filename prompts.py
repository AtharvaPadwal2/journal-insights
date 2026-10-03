EMOTION_PROMPTS = {
    "sadness": "What part of this experience stayed with you the most?",
    "joy": "What made this experience meaningful to you?",
    "love": "What did you appreciate about the connection you described?",
    "anger": "What felt most important to you in this situation?",
    "fear": "What uncertainty would you like to explore further?",
    "surprise": "How did this experience differ from what you expected?",
}

THEME_PROMPTS = {
    "Studies and exams": (
        "What part of your studies would you like to reflect on further?"
    ),
    "Work and deadlines": (
        "What stood out to you about the work situation you described?"
    ),
    "Friends and relationships": (
        "What would you like the people involved to understand?"
    ),
    "Family": (
        "What felt meaningful about the family experience you described?"
    ),
    "Achievement": (
        "What did this experience teach you about your efforts?"
    ),
    "Rest and sleep": (
        "What have you noticed about your rest recently?"
    ),
    "Hobbies": (
        "What did you enjoy or find meaningful about this activity?"
    ),
    "Money": (
        "Which part of the money situation is most on your mind?"
    ),
    "Future plans": (
        "What matters most to you when thinking about your future?"
    ),
}

# More specific matches take priority when both emotion and theme match.
COMBINED_PROMPTS = {
    ("fear", "Studies and exams"): (
        "Which part of the upcoming study or exam experience feels uncertain?"
    ),
    ("joy", "Achievement"): (
        "Which part of your effort would you like to acknowledge?"
    ),
    ("sadness", "Friends and relationships"): (
        "What would you like to express about the connection you described?"
    ),
    ("anger", "Work and deadlines"): (
        "What expectation at work would you like to clarify?"
    ),
}

FALLBACK_PROMPT = "What part of this experience would you like to explore?"


def get_reflection_question(emotion, themes):
    # Prefer an explicit emotion + theme match.
    for item in themes:
        theme = item["theme"]
        question = COMBINED_PROMPTS.get((emotion, theme))

        if question:
            return {
                "question": question,
                "matched_by": "emotion_and_theme",
                "theme": theme,
            }

    # Otherwise choose the theme with the most matched phrases.
    if themes:
        strongest_theme = max(
            themes,
            key=lambda item: len(item.get("matched_phrases", [])),
        )["theme"]

        question = THEME_PROMPTS.get(strongest_theme)

        if question:
            return {
                "question": question,
                "matched_by": "theme",
                "theme": strongest_theme,
            }

    return {
        "question": EMOTION_PROMPTS.get(emotion, FALLBACK_PROMPT),
        "matched_by": (
            "emotion" if emotion in EMOTION_PROMPTS else "fallback"
        ),
        "theme": None,
    }


def build_interpretation(overall, sentiment, sentences, themes):
    summary = [
        f"The emotion model predicts {overall['emotion']} "
        "for the entry as a whole.",
        f"The separate RoBERTa model labels its overall sentiment "
        f"as {sentiment['label']}.",
    ]

    if sentences:
        sentence_labels = list(dict.fromkeys(
            item["emotion"] for item in sentences
        ))

        if len(sentence_labels) > 1:
            summary.append(
                "Sentence predictions include "
                + ", ".join(sentence_labels)
                + ", showing variation in the model's labels across the text."
            )

    if themes:
        summary.append(
            "Keyword-based topic matches include "
            + ", ".join(item["theme"] for item in themes)
            + "."
        )

    return " ".join(summary)