from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

SEED = 42


def build_models():
    logistic = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)),
    ])
    svm = Pipeline([
        ("scale", RobustScaler()),
        ("model", SVC(kernel="rbf", C=2.0, gamma="scale", probability=True,
                      class_weight="balanced", random_state=SEED)),
    ])
    extra_trees = ExtraTreesClassifier(
        n_estimators=450,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    ensemble = VotingClassifier(
        estimators=[("lr", logistic), ("svm", svm), ("et", extra_trees)],
        voting="soft",
        weights=[1.0, 1.2, 1.4],
        n_jobs=-1,
    )
    return {
        "Logistic Regression": logistic,
        "RBF SVM": svm,
        "Extra Trees": extra_trees,
        "Soft Voting Ensemble": ensemble,
    }


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
    }
    return metrics, pred, prob


def main():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    results = []
    best = None
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        metrics, pred, prob = evaluate(name, model, X_test, y_test)
        results.append(metrics)
        if best is None or metrics["roc_auc"] > best["metrics"]["roc_auc"]:
            best = {"name": name, "model": model, "metrics": metrics, "pred": pred, "prob": prob}

    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    joblib.dump(best["model"], out / "best_disease_model.joblib")
    (out / "metrics.json").write_text(json.dumps(results, indent=2))

    schema = {
        c: {"min": float(X[c].min()), "max": float(X[c].max()), "default": float(X[c].median())}
        for c in X.columns
    }
    (out / "feature_schema.json").write_text(json.dumps(schema, indent=2))
    (out / "meta.json").write_text(json.dumps({
        "features": list(X.columns),
        "target_names": list(data.target_names),
        "best_model": best["name"],
    }, indent=2))

    fpr, tpr, _ = roc_curve(y_test, best["prob"])
    fig = plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"{best['name']} AUC={best['metrics']['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Disease Prediction ROC Curve")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out / "roc_curve.png", dpi=180)
    plt.close(fig)

    cm = confusion_matrix(y_test, best["pred"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=data.target_names).plot(ax=ax)
    plt.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    print(pd.DataFrame(results).sort_values("roc_auc", ascending=False).to_string(index=False))
    print(f"\nBest model: {best['name']}")


if __name__ == "__main__":
    main()
