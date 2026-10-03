import numpy as np


def get_model_clues(text, pipeline, top_n=6):
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    features = vectorizer.transform([text])
    predicted_class = classifier.predict(features)[0]

    class_index = int(
        np.flatnonzero(classifier.classes_ == predicted_class)[0]
    )

    # Binary classifiers store one coefficient row.
    if len(classifier.classes_) == 2:
        coefficients = classifier.coef_[0]
        if class_index == 0:
            coefficients = -coefficients
    else:
        coefficients = classifier.coef_[class_index]

    feature_names = vectorizer.get_feature_names_out()
    clues = []

    # Only inspect features present in this input.
    for feature_index, feature_value in zip(
        features.indices, features.data
    ):
        contribution = float(
            feature_value * coefficients[feature_index]
        )

        if contribution > 0:
            clues.append({
                "phrase": str(feature_names[feature_index]),
                "contribution": contribution,
            })

    clues.sort(
        key=lambda clue: clue["contribution"],
        reverse=True,
    )

    return clues[:top_n]