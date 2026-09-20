import time
import traceback
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.task5 import (
    train_model_return_scores_clamp,
    train_model_return_scores_unsw,
    train_model_return_scores_phiusiil,
)

DATA_DIR = Path("task5")

# (display name, keyword in the CSV file name, train function, passing AUC)
DATASETS = [
    ("ClaMP",    "clamp",    train_model_return_scores_clamp,    0.90),
    ("UNSW",     "unsw",     train_model_return_scores_unsw,     0.76),
    ("PhiUSIIL", "phiusiil", train_model_return_scores_phiusiil, 0.85),
]


def find_csv(keyword):
    matches = sorted(p for p in DATA_DIR.glob("*.csv") if keyword in p.name.lower())
    return matches[0] if matches else None


results = {}

for name, keyword, train_fn, target in DATASETS:
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60, flush=True)

    try:
        csv_path = find_csv(keyword)
        if csv_path is None:
            print("Could not find a CSV containing '" + keyword + "' in", DATA_DIR)
            print("CSV files found:", [p.name for p in DATA_DIR.glob("*.csv")])
            results[name] = "NO CSV FOUND"
            continue
        print("Loading", csv_path, flush=True)

        # Load the labeled dataset.
        full_df = pd.read_csv(csv_path)

        # Split the dataset while maintaining the class balance.
        train_df, labeled_test_df = train_test_split(
            full_df,
            test_size=0.20,
            random_state=0,
            stratify=full_df["label"]
        )

        # Save the correct test labels for evaluating our predictions.
        true_test_labels = labeled_test_df["label"].copy()

        # Remove the label column to imitate the autograder.
        test_df = labeled_test_df.drop(columns=["label"])

        # Train the model and predict probabilities.
        print("Training...", flush=True)
        start = time.time()
        test_scores = train_fn(train_df, test_df)
        print("Train + predict took", round(time.time() - start), "seconds")

        # Check that the returned DataFrame has the required structure.
        assert len(test_scores) == len(test_df)
        assert list(test_scores.columns) == ["index", "prob_label_1"]
        assert test_scores["index"].tolist() == test_df.index.tolist()
        assert test_scores["prob_label_1"].between(0, 1).all()

        # Calculate ROC AUC using the hidden correct answers.
        auc_score = roc_auc_score(true_test_labels, test_scores["prob_label_1"])

        print(test_scores.head())
        print()
        print("Prediction rows:", len(test_scores))
        print("ROC AUC:", round(auc_score, 4))

        if auc_score >= target:
            print("PASS: ROC AUC is at least", target)
            results[name] = "PASS (" + str(round(auc_score, 4)) + ")"
        else:
            print("FAIL: ROC AUC is below", target)
            results[name] = "FAIL (" + str(round(auc_score, 4)) + ")"

    except Exception:
        traceback.print_exc()
        results[name] = "ERROR"

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for name, result in results.items():
    print(name + ":", result)