from sklearn.feature_extraction.text import TfidfVectorizer


def create_url_vectorizer():
    """Convert URLs into character-level TF-IDF features."""
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=30000,
        min_df=2,
        lowercase=True,
        sublinear_tf=True,
        dtype="float32"
    )