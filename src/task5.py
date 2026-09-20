import re
import math
from collections import Counter
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ClaMP

def document_hyperparameter_tuning_clamp(train_df, test_df):
    hyperparameters = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "liblinear",
        "class_weight": "balanced",
        "max_iter": 5000,
        "random_state": 0
    }

    return hyperparameters


def train_model_return_scores_clamp(train_df, test_df) -> pd.DataFrame:
    # Separate training features from labels.
    train_features = train_df.drop(columns=["label"])
    train_labels = train_df["label"]

    # Match the test columns to the training columns.
    test_features = test_df[train_features.columns]

    # Fill missing values.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputed_train_features = imputer.fit_transform(train_features)
    imputed_test_features = imputer.transform(test_features)

    # Scale the imputed features.
    scaler = StandardScaler()
    scaled_train_features = scaler.fit_transform(imputed_train_features)
    scaled_test_features = scaler.transform(imputed_test_features)

    # Get Logistic Regression settings.
    hyperparameters = document_hyperparameter_tuning_clamp(train_df, test_df)

    # Create and train the model.
    model = LogisticRegression(**hyperparameters)
    model.fit(scaled_train_features, train_labels)

    # Predict malware probabilities.
    malware_probabilities = model.predict_proba(scaled_test_features)[:, 1]

    # Create the required output DataFrame.
    test_scores = pd.DataFrame({
        "index": test_df.index,
        "prob_label_1": malware_probabilities
    })

    return test_scores



# UNSW-NB15

def document_hyperparameter_tuning_unsw(train_df, test_df):

    hyperparameters = {
        "n_estimators": 400,
        "learning_rate": 0.02,
        "max_depth": 7,
        "min_samples_leaf": 1,
        "subsample": 0.8,
        "random_state": 0,
    }
    return hyperparameters


def train_model_return_scores_unsw(train_df, test_df) -> pd.DataFrame:
    drop_cols = ["label"] + [c for c in train_df.columns if c.startswith("Unnamed") or c == "id"]
    train_features = train_df.drop(columns=drop_cols)
    train_labels = train_df["label"]
    test_features = test_df.reindex(columns=train_features.columns)

    cat_cols = train_features.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in train_features.columns if c not in cat_cols]

    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", keep_empty_features=True), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]), cat_cols),
    ])


    hyperparameters = document_hyperparameter_tuning_unsw(train_df, test_df)
    model = Pipeline([
        ("prep", preprocessor),
        ("gb", GradientBoostingClassifier(**hyperparameters)),
    ])
    model.fit(train_features, train_labels)

    probabilities = model.predict_proba(test_features)[:, 1]
    return pd.DataFrame({"index": test_df.index, "prob_label_1": probabilities})



# PhiUSIIL

def extract_url_features(urls: pd.Series) -> pd.DataFrame:
    """Turn a Series of raw URL strings into a numeric feature table (no internet needed)."""
    suspicious_words = ["login", "signin", "secure", "account", "update", "verify", "bank",
                        "paypal", "confirm", "password", "webscr", "free", "bonus", "lucky"]
    rows = []
    for url in urls.fillna("").astype(str):
        u = url.strip()
        parsed = urlparse(u if "://" in u else "http://" + u)
        host = parsed.netloc.split("@")[-1].split(":")[0].lower()
        host_parts = [p for p in host.split(".") if p]
        tld = host_parts[-1] if host_parts else ""
        n = max(len(u), 1)
        counts = Counter(u)
        entropy = -sum((c / n) * math.log2(c / n) for c in counts.values()) if u else 0.0
        digits = sum(ch.isdigit() for ch in u)
        letters = sum(ch.isalpha() for ch in u)
        rows.append({
            "url_len": len(u),
            "host_len": len(host),
            "path_len": len(parsed.path),
            "query_len": len(parsed.query),
            "n_dots": u.count("."),
            "n_hyphens": u.count("-"),
            "n_underscores": u.count("_"),
            "n_slashes": u.count("/"),
            "n_qmarks": u.count("?"),
            "n_equals": u.count("="),
            "n_ampersands": u.count("&"),
            "n_at": u.count("@"),
            "n_percent": u.count("%"),
            "n_digits": digits,
            "digit_ratio": digits / n,
            "letter_ratio": letters / n,
            "special_ratio": (len(u) - digits - letters) / n,
            "host_digits": sum(ch.isdigit() for ch in host),
            "host_hyphens": host.count("-"),
            "n_subdomains": max(len(host_parts) - 2, 0),
            "tld_len": len(tld),
            "common_tld": int(tld in {"com", "org", "net", "edu", "gov"}),
            "has_ip_host": int(bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))),
            "is_https": int(parsed.scheme == "https"),
            "has_www": int(host.startswith("www.")),
            "has_port": int(":" in parsed.netloc.split("@")[-1]),
            "path_depth": parsed.path.count("/"),
            "double_slash_in_path": int("//" in parsed.path),
            "n_suspicious_words": sum(w in u.lower() for w in suspicious_words),
            "longest_host_token": max((len(p) for p in host_parts), default=0),
            "entropy": entropy,
        })
    return pd.DataFrame(rows, index=urls.index)


def document_hyperparameter_tuning_phiusiil(train_df, test_df):
    hyperparameters = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 0.8,
        "random_state": 0,
    }
    return hyperparameters


def train_model_return_scores_phiusiil(train_df, test_df) -> pd.DataFrame:
    url_col = "URL" if "URL" in train_df.columns else [c for c in train_df.columns if c != "label"][0]
    train_features = extract_url_features(train_df[url_col])
    train_labels = train_df["label"]
    test_features = extract_url_features(test_df[url_col])

    # REPLACE with the output of document_hyperparameter_tuning_phiusiil after running it locally
    hyperparameters = document_hyperparameter_tuning_phiusiil(train_df, test_df)
    model = GradientBoostingClassifier(**hyperparameters)
    model.fit(train_features, train_labels)

    probabilities = model.predict_proba(test_features)[:, 1]
    return pd.DataFrame({"index": test_df.index, "prob_label_1": probabilities})