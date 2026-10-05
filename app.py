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


model_options = {
    "LogReg · baseline": "logreg",
    "Hartmann · includes neutral": "hartmann",
}

selected_model = st.selectbox(
    "Emotion model",
    options=list(model_options),
    key="emotion_model_selection",
    on_change=clear_analysis,
)

emotion_model = model_options[selected_model]
def clear_entry():
    st.session_state["journal_text"] = ""
    st.session_state.pop("analysis", None)


st.text_area(
    "What’s on your mind?",
    placeholder="Write about your day...",
    height=200,
    max_chars=5000,
    key="journal_text",
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
    st.session_state.pop("analysis", None)

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
    else:
        if result.get("model_clues_available", True):
         st.info("No positive model clues found for this entry.")
else:
    st.info("Word contribution explanations are available for LogReg only.")

    st.subheader("Themes and keywords")
    st.caption("Themes use keyword matching; keywords use word frequency.")

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
            st.write(
                f"Sentence {index} · "
                f"{sentence['emotion'].title()}"
            )
            st.text(sentence["sentence"])
    st.subheader("A moment to reflect")

    reflection = result.get("reflection", {})

    if reflection.get("question"):
        with st.container(border=True):
            st.write(reflection["question"])

        st.caption("A curated reflection prompt, not treatment advice.")
    with st.expander("View emotion model scores"):
        st.caption(
            "Uncalibrated model scores—not emotional percentages. "
            "This classifier predicts one emotion label per text."
        )
        st.bar_chart(result["overall"]["scores"])

    st.caption(
        f"Analysis latency: {result['latency_ms']:.2f} ms"
    )

    for note in result["notes"]:
        st.caption(note)

    st.caption(
        "This analyzes text; it does not diagnose mental-health conditions."
    )