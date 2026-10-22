from flask import Flask, render_template, request
from urllib.parse import urlparse
import os
import joblib

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "phishing_model.joblib"
)

model = None

try:
    if os.path.isfile(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Trained phishing detection model loaded.")
    else:
        print("Model not found. Run train_model.py first.")

except Exception as e:
    print("Could not load model:", e)


def validate_url(url):
    if not url:
        return False, "Please enter a URL."

    if any(char.isspace() for char in url):
        return False, "URL must not contain spaces."

    try:
        parsed_url = urlparse(url)

        if parsed_url.scheme not in ("http", "https"):
            return False, "URL must start with http:// or https://."

        if not parsed_url.hostname:
            return False, "Please enter a valid website URL."

        return True, ""

    except ValueError:
        return False, "Invalid URL. Please check and try again."


@app.route("/", methods=["GET", "POST"])
def home():
    url = ""
    message = None
    error = None

    if request.method == "POST":
        url = request.form.get("url", "").strip()

        is_valid, validation_error = validate_url(url)

        if not is_valid:
            error = validation_error

        elif model is None:
            error = (
                "The trained model is unavailable. "
                "Please run train_model.py first."
            )

        else:
            try:
                prediction = int(model.predict([url])[0])

                # In this dataset, label 1 means legitimate
                # and label 0 means phishing.
                if prediction == 1:
                    message = (
                        "Likely LEGITIMATE: the model classified "
                        "this URL as legitimate."
                    )
                else:
                    message = (
                        "Potential PHISHING: the model classified "
                        "this URL as phishing."
                    )

            except Exception as e:
                print("Prediction error:", e)
                error = "Unable to predict this URL. Check the backend."

    return render_template(
        "index.html",
        url=url,
        message=message,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)