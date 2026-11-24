"""
Chronic Kidney Disease (CKD) Prediction using Bagging and Boosting algorithms.

Dataset : UCI "Chronic Kidney Disease" (400 patients, 24 features + class).
Bagging : BaggingClassifier, RandomForest, ExtraTrees
Boosting: AdaBoost, GradientBoosting, XGBoost (optional), LightGBM (optional)
Baseline: single Decision Tree, Logistic Regression

Usage:
    pip install scikit-learn pandas numpy matplotlib seaborn ucimlrepo xgboost lightgbm
    python ckd_prediction.py                       # downloads data via ucimlrepo
    python ckd_prediction.py --csv kidney_disease.csv   # or use a local CSV (Kaggle/UCI copy)
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEED = 42

NUMERIC = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod",
           "pot", "hemo", "pcv", "wbcc", "rbcc"]
CATEGORICAL = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet",
               "pe", "ane"]


# --------------------------------------------------------------------------
# 1. Data loading & cleaning
# --------------------------------------------------------------------------
def load_data(csv_path=None):
    if csv_path:
        df = pd.read_csv(csv_path)
        df = df.drop(columns=[c for c in ["id"] if c in df.columns])
        # Kaggle copy uses these long names -> map to UCI short names
        rename = {"classification": "class", "red_blood_cell_count": "rbcc",
                  "white_blood_cell_count": "wbcc", "packed_cell_volume": "pcv"}
        df = df.rename(columns=rename)
        target = "class"
    else:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=336)
        df = ds.data.features.copy()
        df["class"] = ds.data.targets.iloc[:, 0].values
        target = "class"

    # Clean messy strings: whitespace, tabs, '?' placeholders
    for col in df.select_dtypes(include="object").columns:
        df[col] = (df[col].astype(str).str.strip().str.replace("\t", "")
                   .replace({"?": np.nan, "nan": np.nan, "": np.nan}))

    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fix label / category typos found in the raw data
    df[target] = df[target].replace({"ckd.": "ckd", "notckd.": "notckd"})
    df["dm"] = df["dm"].replace({"yes ": "yes", "no ": "no", "\tno": "no",
                                 "\tyes": "yes"})
    df["cad"] = df["cad"].replace({"\tno": "no"})

    y = (df[target] == "ckd").astype(int)  # 1 = CKD, 0 = not CKD
    X = df.drop(columns=[target])
    return X, y


# --------------------------------------------------------------------------
# 2. Preprocessing + models
# --------------------------------------------------------------------------
def build_preprocessor(X):
    num = [c for c in NUMERIC if c in X.columns]
    cat = [c for c in CATEGORICAL if c in X.columns]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def get_models():
    models = {
        # --- baselines ---
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        # --- bagging family ---
        "Bagging (DT)": BaggingClassifier(
            estimator=DecisionTreeClassifier(), n_estimators=200,
            random_state=SEED, n_jobs=-1),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=SEED, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, random_state=SEED, n_jobs=-1),
        # --- boosting family ---
        "AdaBoost": AdaBoostClassifier(n_estimators=200, learning_rate=0.5,
                                       random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=3, random_state=SEED),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=4,
            eval_metric="logloss", random_state=SEED, n_jobs=-1)
    except ImportError:
        print("xgboost not installed - skipping XGBoost")
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, random_state=SEED,
            verbose=-1)
    except ImportError:
        print("lightgbm not installed - skipping LightGBM")
    return models


# --------------------------------------------------------------------------
# 3. Evaluation
# --------------------------------------------------------------------------
def main(csv_path=None):
    X, y = load_data(csv_path)
    print(f"Data: {X.shape[0]} patients, {X.shape[1]} features | "
          f"CKD={y.sum()}  Not CKD={(y == 0).sum()}")
    print(f"Missing values: {X.isna().sum().sum()} cells\n")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    pre = build_preprocessor(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    rows, fitted, probs = [], {}, {}
    for name, model in get_models().items():
        pipe = Pipeline([("pre", pre), ("clf", model)])

        # 5-fold CV on training set (imputer/scaler fit inside each fold => no leakage)
        cvres = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=scoring)

        # Hold-out test
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        proba = pipe.predict_proba(X_te)[:, 1]
        fitted[name], probs[name] = pipe, proba

        rows.append({
            "Model": name,
            "Type": ("Bagging" if name in ("Bagging (DT)", "Random Forest", "Extra Trees")
                     else "Boosting" if name in ("AdaBoost", "Gradient Boosting",
                                                 "XGBoost", "LightGBM")
                     else "Baseline"),
            "CV Accuracy": cvres["test_accuracy"].mean(),
            "CV F1": cvres["test_f1"].mean(),
            "Test Accuracy": accuracy_score(y_te, pred),
            "Test Precision": precision_score(y_te, pred),
            "Test Recall": recall_score(y_te, pred),
            "Test F1": f1_score(y_te, pred),
            "Test ROC-AUC": roc_auc_score(y_te, proba),
        })

    results = (pd.DataFrame(rows).sort_values("Test F1", ascending=False)
               .reset_index(drop=True))
    pd.set_option("display.width", 200)
    print(results.round(4).to_string(index=False))
    results.round(4).to_csv("ckd_model_comparison.csv", index=False)

    best = results.loc[0, "Model"]
    print(f"\nBest model on test F1: {best}")
    print(classification_report(y_te, fitted[best].predict(X_te),
                                target_names=["Not CKD", "CKD"]))

    # ---- Plots ----
    # (a) metric comparison
    results.set_index("Model")[["Test Accuracy", "Test Recall", "Test F1",
                                "Test ROC-AUC"]].plot.bar(figsize=(11, 5))
    plt.ylim(0.9, 1.01); plt.title("Bagging vs Boosting - CKD prediction")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150); plt.close()

    # (b) ROC curves
    plt.figure(figsize=(6, 5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        plt.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_te, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150); plt.close()

    # (c) confusion matrix of best model
    ConfusionMatrixDisplay.from_estimator(
        fitted[best], X_te, y_te, display_labels=["Not CKD", "CKD"],
        cmap="Blues")
    plt.title(f"Confusion matrix - {best}"); plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150); plt.close()

    # (d) feature importance (best tree-based model)
    clf = fitted[best].named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        names = fitted[best].named_steps["pre"].get_feature_names_out()
        imp = (pd.Series(clf.feature_importances_, index=names)
               .sort_values(ascending=False).head(15))
        plt.figure(figsize=(7, 5))
        sns.barplot(x=imp.values, y=[n.split("__")[-1] for n in imp.index],
                    color="steelblue")
        plt.title(f"Top 15 features - {best}"); plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=150); plt.close()

    print("\nSaved: ckd_model_comparison.csv, model_comparison.png, "
          "roc_curves.png, confusion_matrix.png, feature_importance.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="Path to local CKD csv")
    main(ap.parse_args().csv)
