from pathlib import Path
import os

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
REPO_DIR = PROJECT_DIR.parent

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ Tourism Package Prediction")
st.write(
    "Use the committed machine-learning model to predict "
    "whether a customer is likely to purchase a tourism package."
)


# --------------------------------------------------
# Locate the committed model
# --------------------------------------------------

def find_model():
    """
    Locate a model committed to the repository.

    Optionally set MODEL_PATH to specify the exact model path.
    """

    configured_path = os.getenv("MODEL_PATH")

    if configured_path:
        model_path = Path(configured_path)

        if not model_path.is_absolute():
            model_path = REPO_DIR / model_path

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Configured model file does not exist: {model_path}"
            )

        return model_path

    search_directories = [
        APP_DIR,
        PROJECT_DIR,
        REPO_DIR,
    ]

    candidates = []

    for directory in search_directories:
        for extension in ("*.joblib", "*.pkl"):
            candidates.extend(directory.rglob(extension))

    # Remove duplicate paths and sort deterministically.
    candidates = sorted(set(candidates))

    if not candidates:
        raise FileNotFoundError(
            "No .pkl or .joblib model was found. "
            "Commit the trained model to the repository or "
            "configure MODEL_PATH."
        )

    if len(candidates) > 1:
        st.warning(
            "Multiple model files were found. "
            "The first sorted file will be selected. "
            "Set MODEL_PATH to choose a specific model."
        )

        for candidate in candidates:
            st.write(f"- {candidate.relative_to(REPO_DIR)}")

    return candidates[0]


# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model(model_path: str):
    """
    Load a trusted, repository-committed model.

    Only load pickle/joblib files from sources you trust.
    """

    return joblib.load(model_path)


try:
    model_path = find_model()
    model = load_model(str(model_path))

    st.success("Model loaded successfully.")
    st.caption(f"Model file: {model_path.name}")

except Exception as error:
    st.error(f"Unable to load the model: {error}")

    st.info(
        "Check that the trained model is committed to GitHub, "
        "the file path is correct, and the required dependencies "
        "are installed."
    )

    st.stop()


# --------------------------------------------------
# Model information
# --------------------------------------------------

feature_names = getattr(model, "feature_names_in_", None)

if feature_names is None and hasattr(model, "named_steps"):
    # Some pipelines expose feature names through their final estimator.
    final_estimator = list(model.named_steps.values())[-1]
    feature_names = getattr(final_estimator, "feature_names_in_", None)

if feature_names is not None:
    feature_names = list(feature_names)

st.subheader("Model information")

col1, col2 = st.columns(2)

with col1:
    st.metric("Input features", len(feature_names) if feature_names else "Not exposed")

with col2:
    st.write("Model type")
    st.write(type(model).__name__)


# --------------------------------------------------
# Prediction interface
# --------------------------------------------------

st.subheader("Make predictions")

tab_single, tab_batch = st.tabs([
    "Single prediction", "Batch predictions (CSV)"
])


# --------------------------------------------------
# Single prediction
# --------------------------------------------------

with tab_single:

    if not feature_names:
        st.warning(
            "This model does not expose its input feature names. "
            "Use batch prediction with a CSV containing the "
            "same features and order used during model training."
        )

    else:
        st.write("Enter the customer details below.")

        with st.form("prediction_form"):
            input_values = {}

            for feature in feature_names:
                input_values[feature] = st.number_input(
                    label=str(feature),
                    value=0.0,
                    format="%.4f",
                    key=f"input_{feature}",
                )

            submitted = st.form_submit_button("Predict")

        if submitted:
            try:
                input_df = pd.DataFrame(
                    [input_values],
                    columns=feature_names,
                )

                prediction = model.predict(input_df)

                st.success("Prediction completed.")

                st.write("### Prediction result")
                st.write(prediction[0])

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_df)

                    st.write("### Prediction probabilities")

                    classes = getattr(
                        model,
                        "classes_",
                        list(range(probabilities.shape[1])),
                    )

                    probability_df = pd.DataFrame({
                        "Class": classes,
                        "Probability": probabilities[0],
                    })

                    st.dataframe(
                        probability_df,
                        use_container_width=True,
                    )

            except Exception as error:
                st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# Batch predictions
# --------------------------------------------------

with tab_batch:
    st.write(
        "Upload a CSV file containing the model's input features. "
        "Do not include the target column unless the model requires it."
    )

    uploaded_file = st.file_uploader(
        "Upload customer data",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)

            st.write("### Uploaded data")
            st.dataframe(input_df.head(), use_container_width=True)

            if feature_names:
                missing = [
                    name for name in feature_names
                    if name not in input_df.columns
                ]

                if missing:
                    st.error(
                        "The CSV is missing required features: "
                        + ", ".join(map(str, missing))
                    )
                    st.stop()

                # Preserve the exact feature order used by the model.
                input_df = input_df[feature_names]

            if st.button("Run batch predictions"):
                predictions = model.predict(input_df)

                results = pd.DataFrame({
                    "Prediction": predictions
                })

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_df)

                    classes = getattr(
                        model,
                        "classes_",
                        range(probabilities.shape[1]),
                    )

                    for index, class_label in enumerate(classes):
                        results[
                            f"Probability_{class_label}"
                        ] = probabilities[:, index]

                st.write("### Prediction results")
                st.dataframe(results, use_container_width=True)

                csv_data = results.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download predictions as CSV",
                    data=csv_data,
                    file_name="tourism_predictions.csv",
                    mime="text/csv",
                )

        except Exception as error:
            st.error(f"Unable to process the CSV: {error}")