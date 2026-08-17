
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

st.set_page_config(
    page_title="Credit Risk Classifier",
    page_icon="🏦",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "model" / "artifacts"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
}

MODEL_NOTES = {
    "Logistic Regression": "Linear baseline. Coefficients are easy to explain to a "
                           "credit committee, which matters in regulated lending.",
    "Decision Tree": "A single tree, depth limited to keep it readable. Prone to "
                     "variance but gives human-readable decision rules.",
    "kNN": "Distance based, so it depends heavily on scaling. One-hot encoding "
           "pushes the data into a sparse high dimensional space, which usually hurts it.",
    "Naive Bayes": "Assumes features are independent given the class. That assumption "
                   "is clearly violated here, but it trains instantly.",
    "Random Forest (Ensemble)": "Many decorrelated trees voting together. Normally the "
                                "strongest on tabular data, at the cost of interpretability.",
}



@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load every fitted pipeline plus the feature metadata."""
    meta_path = ARTIFACT_DIR / "feature_meta.joblib"
    if not meta_path.exists():
        return None, None

    models = {}
    for label, filename in MODEL_FILES.items():
        path = ARTIFACT_DIR / filename
        if path.exists():
            models[label] = joblib.load(path)
    return models, joblib.load(meta_path)


@st.cache_data(show_spinner=False)
def read_csv(uploaded) -> pd.DataFrame:
    return pd.read_csv(uploaded)


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """The six metrics required by the assignment."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }



models, meta = load_artifacts()

st.sidebar.title("Controls")

if models is None or not models:
    st.sidebar.error("No trained models found.")
    st.title("Credit Risk Classifier")
    st.error(
        "The `model/artifacts/` folder is empty. Run `python model/train_models.py` "
        "from the project root first, then reload this page."
    )
    st.stop()

TARGET = meta["target"]
CLASS_LABELS = meta["class_labels"]

uploaded_file = st.sidebar.file_uploader(
    "Upload test data (CSV)",
    type="csv",
    help=f"The file needs the 20 feature columns plus a '{TARGET}' column "
         "with 0 = Good credit and 1 = Bad credit.",
)

use_sample = st.sidebar.checkbox(
    "Use the bundled test_data.csv instead", value=not bool(uploaded_file)
)

selected_model = st.sidebar.selectbox(
    "Choose a model", list(models.keys()), index=0
)

threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help="Probability above which an applicant is flagged as bad credit. "
         "Lowering it catches more defaulters but rejects more good customers.",
)

st.sidebar.caption(
    "Positive class = **Bad credit (1)**, since catching likely defaulters "
    "is the business goal."
)



st.title("🏦 Credit Risk Classifier")
st.write(
    "Five classification models trained on the Statlog (German Credit Data) set, "
    "compared on a held-out test sample."
)

if uploaded_file is not None and not use_sample:
    data = read_csv(uploaded_file)
    source = uploaded_file.name
elif (ROOT / "test_data.csv").exists():
    data = pd.read_csv(ROOT / "test_data.csv")
    source = "test_data.csv (bundled)"
else:
    st.warning("Please upload a test CSV using the sidebar to continue.")
    st.stop()

# Validate the uploaded file before doing anything with it.
missing = [c for c in meta["feature_columns"] if c not in data.columns]
if missing:
    st.error(
        f"These required columns are missing from the file: {', '.join(missing[:8])}"
        + (" ..." if len(missing) > 8 else "")
    )
    st.stop()

if TARGET not in data.columns:
    st.error(
        f"The file has no '{TARGET}' column, so evaluation metrics cannot be "
        "calculated. Please include the true labels."
    )
    st.stop()

X = data[meta["feature_columns"]]
y_true = data[TARGET].astype(int)

st.caption(
    f"Source: **{source}** — {len(data)} rows, {len(meta['feature_columns'])} features. "
    f"Class split: {int((y_true == 0).sum())} good / {int((y_true == 1).sum())} bad."
)



tab_single, tab_compare, tab_preview = st.tabs(
    ["Selected model", "Model Comparision", "Data preview"]
)


with tab_single:
    pipeline = models[selected_model]
    y_prob = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    st.subheader(selected_model)
    st.info(MODEL_NOTES.get(selected_model, ""), icon="ℹ️")

    scores = compute_metrics(y_true, y_pred, y_prob)

    cols = st.columns(6)
    for col, (name, value) in zip(cols, scores.items()):
        col.metric(name, f"{value:.3f}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown("--Confusion matrix--")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=[CLASS_LABELS[0], CLASS_LABELS[1]],
            yticklabels=[CLASS_LABELS[0], CLASS_LABELS[1]],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        tn, fp, fn, tp = cm.ravel()
        st.caption(
            f"{fn} risky applicants were missed (approved but actually bad), "
            f"and {fp} good applicants were wrongly rejected."
        )

    with right:
        st.markdown("--ROC curve--")
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {scores['AUC']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Random")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("--Classification report--")
    report = classification_report(
        y_true,
        y_pred,
        target_names=[CLASS_LABELS[0], CLASS_LABELS[1]],
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).transpose().round(3), width="stretch")


with tab_compare:
    st.subheader("Comparison across all models")

    rows = []
    for label, pipeline in models.items():
        prob = pipeline.predict_proba(X)[:, 1]
        pred = (prob >= threshold).astype(int)
        entry = {"Model": label}
        entry.update(compute_metrics(y_true, pred, prob))
        rows.append(entry)

    comparison = pd.DataFrame(rows).set_index("Model")

    st.dataframe(
        comparison.style.format("{:.3f}").highlight_max(axis=0, color="#d8f0d8"),
        width="stretch",
    )
    st.caption("Best value in each column is highlighted.")

    st.markdown("--Visual comparison--")
    metric_choice = st.radio(
        "Metric to plot",
        list(comparison.columns),
        horizontal=True,
        label_visibility="collapsed",
    )
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ordered = comparison[metric_choice].sort_values()
    ax.barh(ordered.index, ordered.values, color="#4C78A8")
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric_choice)
    for i, v in enumerate(ordered.values):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    winner = comparison["MCC"].idxmax()
    st.success(
        f"Best overall on this test set by MCC: **{winner}** "
        f"(MCC = {comparison.loc[winner, 'MCC']:.3f}). MCC is used as the tie-breaker "
        "because it stays honest on imbalanced data, unlike accuracy."
    )

    st.download_button(
        "Download comparison as CSV",
        comparison.round(4).to_csv().encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
    )


with tab_preview:
    st.subheader("Test data preview")
    st.dataframe(data.head(25), width="stretch")

    st.markdown("**Numeric feature summary**")
    numeric_present = [c for c in meta["numeric_columns"] if c in data.columns]
    st.dataframe(data[numeric_present].describe().T.round(2), width="stretch")

    st.markdown("**Predicted vs actual, per row**")
    pipeline = models[selected_model]
    prob = pipeline.predict_proba(X)[:, 1]
    preview = pd.DataFrame(
        {
            "Actual": y_true.map(CLASS_LABELS),
            "Predicted": pd.Series(
                (prob >= threshold).astype(int), index=y_true.index
            ).map(CLASS_LABELS),
            "P(bad credit)": np.round(prob, 3),
        }
    )
    preview["Correct"] = np.where(preview["Actual"] == preview["Predicted"], "yes", "no")
    st.dataframe(preview.head(50), width="stretch")
