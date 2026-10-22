
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

from feature_extraction import create_url_vectorizer


# File locations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(
    BASE_DIR, "PhiUSIIL_Phishing_URL_Dataset.csv"
)
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "phishing_model.joblib")


def train_model():
    print("Loading dataset...")

    # Load only the URL and target columns
    df = pd.read_csv(
        DATASET_PATH,
        usecols=["URL", "label"]
    )

    # Clean missing values
    df = df.dropna(subset=["URL", "label"])
    df["URL"] = df["URL"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])

    # This implementation expects binary labels 0 and 1
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(
            "Expected the dataset's label column to contain only 0 and 1."
        )

    X = df["URL"]
    y = df["label"].astype(int)

    print("Dataset size:", len(df))
    print("Label counts:")
    print(y.value_counts().sort_index())

    # Split data before learning TF-IDF patterns
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # TF-IDF and classifier are saved together
    model = Pipeline([
        ("tfidf", create_url_vectorizer()),
        ("classifier", SGDClassifier(
            loss="log_loss",
            max_iter=20,
            tol=0.001,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    print("\nTraining character-level TF-IDF model...")
    model.fit(X_train, y_train)

    print("\nEvaluating model...")
    predictions = model.predict(X_test)

    print("Accuracy:", round(
        accuracy_score(y_test, predictions), 4
    ))
    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    # Save the complete trained pipeline
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nModel saved successfully!")
    print("Location:", MODEL_PATH)


if __name__ == "__main__":
    train_model()