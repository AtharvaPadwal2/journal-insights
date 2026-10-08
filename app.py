import streamlit as st

from analyzer import analyze_text

st.set_page_config(
    page_title="Journal Insights",
    page_icon="🌿",
    layout="centered",
)

st.title("🌿 Journal Insights")
st.caption("Explore emotions expressed in your English journal text.")
st.caption("Entries are not saved by this application.")


def clear_analysis():
    st.session_state.pop("analysis", None)


def clear_entry():
    st.session_state["journal_text"] = ""
    clear_analysis()


model_options = {
    "LogReg · baseline": "logreg",
    "SVM · improved baseline": "svm",
    "Hartmann · includes neutral": "hartmann",
    "Experimental · multi-emotion": "weighted",
}

selected_model = st.selectbox(
    "Emotion model",
    options=list(model_options),
    key="emotion_model_selection",
    on_change=clear_analysis,
)

emotion_model = model_options[selected_model]

if emotion_model == "weighted":
    st.caption(
        "Experimental: supports multiple emotion labels. "
        "Journal reliability is still being evaluated."
    )

st.text_area(
    "What’s on your mind?",
    placeholder="Write about your day...",
    height=200,
    max_chars=5000,
    key="journal_text",
    on_change=clear_analysis,
)

analyze_column, clear_column = st.columns(2)

with analyze_column:
    analyze_clicked = st.button(
        "Analyze",
        type="primary",
        use_container_width=True,
    )

with clear_column:
    st.button(
        "Clear",
        on_click=clear_entry,
        use_container_width=True,
    )

if analyze_clicked:
    clear_analysis()

    try:
        with st.spinner("Analyzing your text..."):
            st.session_state["analysis"] = analyze_text(
                st.session_state["journal_text"],
                emotion_model=emotion_model,
            )
    except (ValueError, FileNotFoundError) as error:
        st.warning(str(error))

result = st.session_state.get("analysis")

if result:
    st.divider()
    st.subheader("Your text at a glance")
    st.write(result.get("interpretation", ""))

    emotion_column, sentiment_column = st.columns(2)

    with emotion_column:
        if result["emotion_model"] == "weighted":
            st.write("**Predicted emotions · experimental**")

            emotions = result["overall"]["emotions"]

            if emotions:
                st.write(
                    ", ".join(label.title() for label in emotions)
                )
            else:
                st.write("No labels above threshold")
                st.caption(
                    "This does not establish that the entry is neutral."
                )
        else:
            st.metric(
                "Predicted emotion",
                result["overall"]["emotion"].title(),
            )

    with sentiment_column:
        st.metric(
            "Sentiment · RoBERTa",
            result["sentiment"]["label"].title(),
        )
        st.caption(
            "Sentiment scores are uncalibrated model predictions, "
            "not emotional percentages."
        )

    with st.expander("View sentiment model scores"):
        st.bar_chart(result["sentiment"]["scores"])

    st.subheader("Model clues")
    st.caption(
        "Words and phrases that supported the overall prediction. "
        "These explain model scoring, not psychological causes."
    )

    clues = result.get("model_clues", [])

    if clues:
        st.dataframe(
            clues,
            hide_index=True,
            use_container_width=True,
            column_config={
                "phrase": "Word / phrase",
                "contribution": st.column_config.NumberColumn(
                    "Model contribution",
                    format="%.4f",
                ),
            },
        )
    elif result.get("model_clues_available", True):
        st.info("No positive model clues found for this entry.")
    else:
        st.info(
            "Word contribution explanations are available "
            "for LogReg only."
        )

    st.subheader("Themes and keywords")
    st.caption(
        "Themes use keyword matching; keywords use word frequency."
    )

    topics = result.get("topics", {})
    themes = topics.get("themes", [])
    keywords = topics.get("keywords", [])

    if themes:
        for theme in themes:
            with st.container(border=True):
                st.text(theme["theme"])
                st.text(
                    "Matched phrases: "
                    + ", ".join(theme["matched_phrases"])
                )
    else:
        st.info("No clear theme detected.")

    if keywords:
        st.text(
            "Keywords: "
            + ", ".join(keyword["word"] for keyword in keywords)
        )
    else:
        st.caption("No useful keywords found.")

    st.subheader("Sentence by sentence")

    for index, sentence in enumerate(result["sentences"], start=1):
        with st.container(border=True):
            if result["emotion_model"] == "weighted":
                sentence_emotions = sentence["emotions"]
                label_text = (
                    ", ".join(
                        label.title() for label in sentence_emotions
                    )
                    if sentence_emotions
                    else "No labels above threshold"
                )
            else:
                label_text = sentence["emotion"].title()

            st.write(f"Sentence {index} · {label_text}")
            st.text(sentence["sentence"])
            sentence_sentiment = sentence.get("sentiment")

            if sentence_sentiment:
                st.caption(
                    "Sentence sentiment · RoBERTa: "
                    + sentence_sentiment["label"].title()
                )

    if result["emotion_model"] == "weighted":
        st.caption(
            "Sentence labels and whole-entry labels may differ. "
            "An empty selection does not establish neutrality."
        )

    if result["emotion_model"] == "weighted":
        st.subheader("Emotions selected across sentences")
        st.caption(
            "These are sentence-level model selections, "
            "not a new whole-entry prediction. "
            "They may include incorrect labels."
        )

        emotion_sentences = {}

        for index, sentence in enumerate(
            result["sentences"], start=1
        ):
            for label in sentence["emotions"]:
                emotion_sentences.setdefault(label, []).append(index)

        if emotion_sentences:
            st.dataframe(
                [
                    {
                        "Emotion": label.title(),
                        "Sentence numbers": ", ".join(
                            str(number) for number in numbers
                        ),
                    }
                    for label, numbers in emotion_sentences.items()
                ],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info(
                "No sentence-level labels crossed their thresholds."
            )
    st.subheader("A moment to reflect")

    reflection = result.get("reflection", {})

    if reflection.get("question"):
        with st.container(border=True):
            st.write(reflection["question"])

        st.caption(
            "A curated reflection prompt, not treatment advice."
        )

    with st.expander("View emotion model scores"):
        if result["emotion_model"] == "svm":
            st.caption(
                "SVM scores are decision margins, not probabilities. "
                "Negative values are normal; the highest margin "
                "determines the predicted label."
            )
        elif result["emotion_model"] == "weighted":
            st.caption(
                "Each label has its own selection threshold. "
                "Scores are uncalibrated, not emotion intensities, "
                "and need not sum to one."
            )

            st.dataframe(
                [
                    {
                        "Emotion": label.title(),
                        "Score": score,
                        "Threshold": (
                            result["overall"]["thresholds"][label]
                        ),
                        "Selected": (
                            label in result["overall"]["emotions"]
                        ),
                    }
                    for label, score in (
                        result["overall"]["scores"].items()
                    )
                ],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Score": st.column_config.NumberColumn(
                        "Score",
                        format="%.3f",
                    ),
                    "Threshold": st.column_config.NumberColumn(
                        "Threshold",
                        format="%.2f",
                    ),
                },
            )
        else:
            st.caption(
                "Uncalibrated model scores, not emotional percentages. "
                "This classifier predicts one emotion label per text."
            )

        st.bar_chart(result["overall"]["scores"])

    st.caption(
        f"Analysis latency: {result['latency_ms']:.2f} ms"
    )

    for note in result["notes"]:
        st.caption(note)

    st.caption(
        "This analyzes text; it does not diagnose "
        "mental-health conditions."
    )