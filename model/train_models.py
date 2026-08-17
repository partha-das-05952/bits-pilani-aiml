
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.25
TARGET = "credit_risk"          

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

CANDIDATE_FILES = [DATA_DIR / "german.data", DATA_DIR / "german.csv"]
ARTIFACT_DIR = ROOT / "model" / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


COLUMN_NAMES = [
    "checking_account_status",      
    "duration_months",              
    "credit_history",               
    "purpose",                      
    "credit_amount",                
    "savings_account",              
    "employment_since",             
    "installment_rate_pct",         
    "personal_status_sex",          
    "other_debtors",                
    "residence_since_years",        
    "property_type",                
    "age_years",                    
    "other_installment_plans",      
    "housing",                      
    "existing_credits_count",       
    "job_type",                     
    "dependents_count",             
    "telephone",                    
    "foreign_worker",               
    TARGET,
]

NUMERIC_COLS = [
    "duration_months",
    "credit_amount",
    "installment_rate_pct",
    "residence_since_years",
    "age_years",
    "existing_credits_count",
    "dependents_count",
]
CATEGORICAL_COLS = [c for c in COLUMN_NAMES if c not in NUMERIC_COLS + [TARGET]]



def load_data() -> pd.DataFrame:
    
    raw_file = next((p for p in CANDIDATE_FILES if p.exists()), None)
    if raw_file is None:
        raise FileNotFoundError(
            f"\nCould not find german.data or german.csv inside {DATA_DIR}.\n"
            
        )

    
    first_line = raw_file.read_text(encoding="utf-8").splitlines()[0]
    separator = "," if first_line.count(",") >= 20 else r"\s+"

    df = pd.read_csv(raw_file, sep=separator, header=None, names=COLUMN_NAMES)
    print(f"  file           : {raw_file.name}")

   
    df[TARGET] = df[TARGET].map({1: 0, 2: 1})

    if df[TARGET].isna().any():
        raise ValueError("Unexpected target values found - expected only 1 and 2.")

    return df



def build_preprocessor() -> ColumnTransformer:
    
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLS),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
        ],
        remainder="drop",
    )



def build_models() -> dict:
    
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=15,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "kNN": KNeighborsClassifier(
            n_neighbors=15,
            weights="distance",
        ),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }



def evaluate(model, X, y_true) -> dict:
   
    y_pred = model.predict(X)
    
    y_prob = model.predict_proba(X)[:, 1]

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def misclassification_cost(y_true, y_pred) -> int:

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return int(5 * fn + 1 * fp)



def main() -> None:
    print("Loading dataset ...")
    df = load_data()
    print(f"  shape          : {df.shape[0]} rows x {df.shape[1] - 1} features")
    print(f"  class balance  : {df[TARGET].value_counts().to_dict()}  (0=Good, 1=Bad)")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  train / test   : {len(X_train)} / {len(X_test)}")

    
    test_df = X_test.copy()
    test_df[TARGET] = y_test.values
    test_df.to_csv(ROOT / "test_data.csv", index=False)
    print(f"\nSaved test_data.csv ({len(test_df)} rows)")

    results = []
    print("\nTraining models ...")
    for name, classifier in build_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(X_train, y_train)

        scores = evaluate(pipeline, X_test, y_test)
        scores["Cost"] = misclassification_cost(y_test, pipeline.predict(X_test))
        scores["Model"] = name
        results.append(scores)

        filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(pipeline, ARTIFACT_DIR / f"{filename}.joblib")

        line = "  ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in scores.items()
            if k != "Model"
        )
        print(f"  {name:<26} {line}")

    
    joblib.dump(
        {
            "target": TARGET,
            "feature_columns": list(X.columns),
            "numeric_columns": NUMERIC_COLS,
            "categorical_columns": CATEGORICAL_COLS,
            "class_labels": {0: "Good credit", 1: "Bad credit"},
        },
        ARTIFACT_DIR / "feature_meta.joblib",
    )

    metrics_df = pd.DataFrame(results)[
        ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC", "Cost"]
    ]
    metrics_df.to_csv(ROOT / "model" / "metrics_summary.csv", index=False)

    
    rounded = metrics_df.copy()
    for col in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
        rounded[col] = rounded[col].map(lambda v: f"{v:.3f}")
    (ROOT / "model" / "readme_table.md").write_text(
        rounded.to_markdown(index=False), encoding="utf-8"
    )

    best = metrics_df.loc[metrics_df["MCC"].idxmax(), "Model"]
    cheapest = metrics_df.loc[metrics_df["Cost"].idxmin(), "Model"]
    print(f"\nSaved artifacts to {ARTIFACT_DIR}")
    print(f"Best model by MCC        : {best}")
    print(f"Lowest business cost     : {cheapest}")


if __name__ == "__main__":
    main()
