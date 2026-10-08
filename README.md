# 🧠 Journal Insights AI
### Understanding Emotions Through Artificial Intelligence

**Journal Insights AI** is an experimental AI-powered journal analysis system that uses Natural Language Processing (NLP) and Machine Learning to analyze written journal entries, identify emotions, understand sentiment, recognize recurring themes, and generate meaningful insights.

The project aims to move beyond basic positive and negative sentiment classification by examining the different emotions expressed throughout a journal entry.

The long-term vision is to build a privacy-conscious, emotionally aware AI assistant capable of responding to users with contextual, supportive, and personalized reflections.

> **Project Status:** 🚧 Active Development | AI/ML Mini Project

---

## 📌 Project Overview

People often express multiple emotions within a single journal entry. Traditional sentiment analysis systems may classify an entire passage as positive or negative, overlooking the emotional complexity of the text.

Journal Insights AI addresses this problem through a modular NLP pipeline that combines:

- Emotion classification
- Sentiment analysis
- Sentence-level emotion detection
- Multi-label emotion analysis
- Theme and keyword extraction
- Model evaluation and validation
- Interactive visualization

Rather than assigning a single emotional interpretation, the system attempts to understand how different emotions appear across individual sentences and the complete journal entry.

## ✨ Key Features

### 🎭 Emotion Detection
- Identifies emotions expressed in journal entries.
- Supports whole-entry emotion classification.
- Analyzes emotions at sentence level.
- Explores multi-label emotion predictions.
- Uses trained NLP classification models.

### 💬 Sentiment Analysis
- Classifies text sentiment.
- Identifies positive, negative, and neutral expressions.
- Supports contextual sentiment analysis using RoBERTa.
- Provides sentiment predictions for individual sentences.

### 🧩 Theme and Keyword Extraction
- Identifies common themes within journal entries.
- Extracts frequently occurring relevant keywords.
- Uses rule-based theme identification.
- Helps organize recurring discussion topics.

### 📊 Interactive Analysis Dashboard
- Built using Streamlit.
- Accepts user-written journal entries.
- Displays predicted emotions and sentiment.
- Visualizes emotion model outputs.
- Presents sentence-level analysis.
- Displays detected themes and keywords.

### 🧪 Model Evaluation
- Uses separate training, validation, and testing datasets.
- Evaluates classification performance.
- Tracks accuracy, precision, recall, and F1-score.
- Compares alternative model configurations.
- Includes experiments with human-annotated journal entries.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Natural Language Processing | NLTK, Transformers |
| Emotion Classification | Linear Classification Models |
| Sentiment Analysis | VADER, RoBERTa |
| Data Processing | Pandas, NumPy |
| Dataset Management | Hugging Face Datasets |
| User Interface | Streamlit |
| Visualization | Streamlit Charts |
| Version Control | Git & GitHub |

---

## 🏗️ System Architecture

```text
             USER JOURNAL ENTRY
                     |
                     v
              TEXT PREPROCESSING
                     |
          +----------+----------+
          |          |          |
          v          v          v
       EMOTION    SENTIMENT   KEYWORDS
       ANALYSIS   ANALYSIS    & THEMES
          |          |          |
          v          v          v
      Sentence    Positive/   Recurring
      Emotions    Negative/   Topics
                  Neutral
          |          |          |
          +----------+----------+
                     |
                     v
              RESULT AGGREGATION
                     |
                     v
              STREAMLIT DASHBOARD
                     |
                     v
         EMOTION & SENTIMENT INSIGHTS
```

**Future Architecture**

```text
Journal Entry
     |
     v
NLP Analysis Pipeline
     |
     v
Structured Emotion Data
     |
     v
Context-Aware Language Model
     |
     v
Personalized Supportive Reflection
```

---

## 📂 Datasets

The project uses publicly available datasets to train and evaluate its emotion classification models.

### 1. DAIR-AI Emotion Dataset

Source: [Hugging Face — dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)

Contains text samples classified into six emotional categories.

| Emotion | Label |
|---|---|
| Sadness | 0 |
| Joy | 1 |
| Love | 2 |
| Anger | 3 |
| Fear | 4 |
| Surprise | 5 |

### 2. Extended Emotion Datasets

Additional experiments explore fine-grained emotional categories, including optimism, pessimism, and other contextual emotions.

The project also includes experiments with journal-specific validation data to assess how models trained on public datasets behave on more complex personal writing.

**Dataset preprocessing includes:**
- Removing duplicate records.
- Checking conflicting labels.
- Cleaning invalid or empty entries.
- Separating training and evaluation data.
- Preventing cross-split data leakage.

---

## 🧠 Machine Learning Approach

Journal Insights AI follows a modular machine learning approach.

### Step 1: Text Preprocessing
Raw journal entries undergo basic cleaning and text preparation before analysis.

### Step 2: Emotion Classification
Machine learning models identify emotional patterns from text using vectorized textual features or contextual NLP representations.

### Step 3: Sentence-Level Analysis
Longer journal entries are divided into sentences, allowing different emotions to be identified across the passage.

### Step 4: Sentiment Detection
Sentiment models analyze the emotional polarity of the text.

### Step 5: Theme Extraction
Rule-based matching identifies relevant topics and recurring themes.

### Step 6: Result Interpretation
Predictions are organized into understandable outputs and presented through an interactive dashboard.

---

## 📊 Model Performance

### Initial Emotion Classification Baseline

An early experiment used a Scikit-learn text classification pipeline on the DAIR-AI Emotion dataset.

| Metric | Validation Result |
|---|---:|
| Accuracy | 83.30% |
| Macro F1-Score | 75.54% |
| Training Time | 3.24 seconds |

These values represent an initial experimental baseline rather than the final performance of the evolving system.

Later model comparisons and dataset experiments are intended to improve performance on complex, multi-emotion journal entries.

---

## 🖥️ Installation and Setup

### Prerequisites

- Python 3.13 (recommended development environment)
- pip
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/journal-insights.git
cd journal-insights
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

**Windows:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will start locally, typically at:

```text
http://localhost:8501
```

**Note:** The trained model artifacts and any required NLP resources must be available before running inference. Depending on the repository configuration, local model training or resource downloads may be necessary.

---

## 📝 Example Analysis

**Sample Journal Entry:**

> I was afraid my proposal would fail while I waited for the decision. When it was approved, I felt surprised and happy. Now I feel hopeful about tomorrow.

**Illustrative Analysis:**

| Sentence | Detected Emotion |
|---|---|
| I was afraid my proposal would fail. | Fear, Pessimism |
| When it was approved, I felt surprised and happy. | Joy, Surprise, Love, Optimism |
| Now I feel hopeful about tomorrow. | Optimism |

**Sentiment Progression:**

Negative → Positive → Positive

This example demonstrates why analyzing individual sentences can be more informative than relying entirely on a single emotion label for the full entry.

Predictions are experimental and may not fully reflect the author's intended emotions.

---

## 🔬 Evaluation and Experimentation

The project includes experiments focused on improving NLP performance for journal-style writing.

Research areas include:

- Comparing emotion classification approaches.
- Evaluating binary and multi-label predictions.
- Handling entries containing conflicting emotions.
- Testing models on human-annotated examples.
- Investigating label imbalance.
- Comparing classification thresholds.
- Improving journal-specific model performance.

The goal is to develop an analysis pipeline that performs consistently on realistic journal entries, rather than relying only on conventional benchmark accuracy.

---

## 🗺️ Project Roadmap

### Phase 1 — Core NLP Pipeline
- [x] Set up Python development environment.
- [x] Load initial emotion dataset.
- [x] Perform dataset cleaning.
- [x] Train baseline emotion classifier.
- [x] Evaluate validation performance.
- [x] Implement sentiment classification.
- [x] Add theme and keyword extraction.
- [x] Build Streamlit interface.

### Phase 2 — Advanced Emotion Intelligence
- [x] Experiment with fine-grained emotion labels.
- [x] Add contextual sentiment analysis.
- [x] Explore sentence-level multi-label predictions.
- [ ] Improve performance on mixed-emotion entries.
- [ ] Expand human-annotated validation.
- [ ] Calibrate model thresholds.
- [ ] Improve emotion aggregation.

### Phase 3 — Intelligent Reflection System
- [ ] Develop context-aware response generation.
- [ ] Convert predictions into structured AI context.
- [ ] Generate supportive natural-language reflections.
- [ ] Improve recognition of emotional transitions.
- [ ] Evaluate response relevance and safety.
- [ ] Implement safeguards for sensitive journal entries.

### Phase 4 — Future Enhancements
- [ ] Integrate Plutchik's Wheel of Emotions.
- [ ] Introduce advanced emotion visualizations.
- [ ] Generate downloadable analysis reports.
- [ ] Explore privacy-preserving personalization.
- [ ] Optimize models for efficient inference.
- [ ] Explore integration with Project Unwind.

---

## 🔗 Future Integration — Project Unwind

Journal Insights AI is being developed as an independent academic project, with potential future integration into **Project Unwind**, a privacy-focused mental wellness platform.

Potential integration features include:

- Emotion-aware journal analysis.
- Contextual AI reflections.
- Optional emotional trend visualization.
- More informative journal insights.
- Supportive and non-judgmental language generation.

Integration is a future objective and is not part of the current standalone implementation.

---

## 🔐 Privacy and Responsible AI

Journal Insights AI is designed with privacy and responsible AI development in mind.

- The initial Streamlit prototype does not intentionally save journal entries through its application workflow.
- Model outputs represent predictions, not psychological facts.
- Emotion classifications can be incomplete or incorrect.
- The system is not intended to diagnose mental health conditions.
- AI-generated reflections, when implemented, will require additional safety evaluation.
- Sensitive content should not be shared with external model providers without appropriate disclosure and user consent.

### Disclaimer

**Journal Insights AI is an experimental educational project, not a medical or clinical tool.**

Its outputs should not be interpreted as diagnoses, professional psychological assessments, or substitutes for qualified mental health support.

---

## 👨‍💻 Development

**Atharva Padwal**

IT Engineering Student | Full-Stack Developer | AI/ML Enthusiast

**Areas of Work:**
- Machine Learning
- Natural Language Processing
- Model Training and Evaluation
- Python Development
- Data Preprocessing
- Interactive Application Development

---

## 📄 License

License information will be added to the repository.

---

### ⭐ Project Vision

*To transform journal entries into meaningful emotional insights through machine learning, while prioritizing user privacy, responsible AI, and supportive human-centered interactions.*

**Journal Insights AI — Understanding the emotions behind the words.**
