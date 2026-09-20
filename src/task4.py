import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
class ModelMetrics:
    def __init__(self, model_type:str,train_metrics:dict,test_metrics:dict,feature_importance_df:pd.DataFrame):
        self.model_type = model_type
        self.train_metrics = train_metrics
        self.test_metrics = test_metrics
        self.feat_imp_df = feature_importance_df
        self.feat_name_col = "Feature"
        self.imp_col = "Importance"
    
    def add_train_metric(self,metric_name:str,metric_val:float):
        self.train_metrics[metric_name] = metric_val

    def add_test_metric(self,metric_name:str,metric_val:float):
        self.test_metrics[metric_name] = metric_val

    def __str__(self): 
        output_str = f"MODEL TYPE: {self.model_type}\n"
        output_str += f"TRAINING METRICS:\n"
        for key in sorted(self.train_metrics):
            output_str += f"  - {key} : {self.train_metrics[key]:.4f}\n"
        output_str += f"TESTING METRICS:\n"
        for key in sorted(self.test_metrics):
            output_str += f"  - {key} : {self.test_metrics[key]:.4f}\n"
        if self.feat_imp_df is not None:
            output_str += f"FEATURE IMPORTANCES:\n"
            for i in self.feat_imp_df.index:
                output_str += f"  - {self.feat_imp_df[self.feat_name_col][i]} : {self.feat_imp_df[self.imp_col][i]:.4f}\n"
        return output_str

def calculate_naive_metrics(train_features:pd.DataFrame, test_features:pd.DataFrame, train_targets:pd.Series, test_targets:pd.Series, naive_assumption:int) -> ModelMetrics:
    # TODO: Read the function description in
    # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task4.html
    # and implement the function as described.

# 1. Create prediction arrays filled with naive_assumption matching target lengths
    train_preds = np.full(len(train_targets), naive_assumption)
    test_preds = np.full(len(test_targets), naive_assumption)

    # 2. Compute training metrics (y_true, y_pred) rounded to 4 decimals
    train_accuracy = round(accuracy_score(train_targets, train_preds), 4)
    train_precision = round(precision_score(train_targets, train_preds, zero_division=0), 4)
    train_recall = round(recall_score(train_targets, train_preds, zero_division=0), 4)
    train_f1 = round(f1_score(train_targets, train_preds, zero_division=0), 4)

    # 3. Compute testing metrics (y_true, y_pred) rounded to 4 decimals
    test_accuracy = round(accuracy_score(test_targets, test_preds), 4)
    test_precision = round(precision_score(test_targets, test_preds, zero_division=0), 4)
    test_recall = round(recall_score(test_targets, test_preds, zero_division=0), 4)
    test_f1 = round(f1_score(test_targets, test_preds, zero_division=0), 4)

    # 4. Populate dictionaries with numerical results
    train_metrics = {
        "accuracy": train_accuracy,
        "recall": train_recall,
        "precision": train_precision,
        "fscore": train_f1
    }
    test_metrics = {
        "accuracy": test_accuracy,
        "recall": test_recall,
        "precision": test_precision,
        "fscore": test_f1
    }

    # 5. Return completed ModelMetrics instance
    naive_metrics = ModelMetrics("Naive", train_metrics, test_metrics, None)
    return naive_metrics



def calculate_logistic_regression_metrics(train_features:pd.DataFrame, test_features:pd.DataFrame, train_targets:pd.Series, test_targets:pd.Series, n_feat_importance:int, logreg_kwargs) -> tuple[ModelMetrics,LogisticRegression]:
    # TODO: Read the function description in
    # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task4.html


    model = LogisticRegression(**logreg_kwargs)
    model.fit(train_features, train_targets)

    # Predictions and probabilities
    train_preds = model.predict(train_features)
    test_preds = model.predict(test_features)
    train_probs = model.predict_proba(train_features)[:, 1]
    test_probs = model.predict_proba(test_features)[:, 1]

    # Confusion matrices for FPR / FNR
    tn, fp, fn, tp = confusion_matrix(train_targets, train_preds).ravel()
    train_fpr = fp / (fp + tn)
    train_fnr = fn / (fn + tp)
    tn, fp, fn, tp = confusion_matrix(test_targets, test_preds).ravel()
    test_fpr = fp / (fp + tn)
    test_fnr = fn / (fn + tp)

    train_metrics = {
        "accuracy":  round(accuracy_score(train_targets, train_preds), 4),
        "recall":    round(recall_score(train_targets, train_preds), 4),
        "precision": round(precision_score(train_targets, train_preds), 4),
        "fscore":    round(f1_score(train_targets, train_preds), 4),
        "fpr":       round(train_fpr, 4),
        "fnr":       round(train_fnr, 4),
        "roc_auc":   round(roc_auc_score(train_targets, train_probs), 4),
    }
    test_metrics = {
        "accuracy":  round(accuracy_score(test_targets, test_preds), 4),
        "recall":    round(recall_score(test_targets, test_preds), 4),
        "precision": round(precision_score(test_targets, test_preds), 4),
        "fscore":    round(f1_score(test_targets, test_preds), 4),
        "fpr":       round(test_fpr, 4),
        "fnr":       round(test_fnr, 4),
        "roc_auc":   round(roc_auc_score(test_targets, test_probs), 4),
    }

    # RFE to pick the top N features
    rfe = RFE(estimator=LogisticRegression(**logreg_kwargs), n_features_to_select=n_feat_importance)
    rfe.fit(train_features, train_targets)
    top_features = train_features.columns[rfe.support_]

    # Retrain on only the selected features
    rfe_model = LogisticRegression(**logreg_kwargs)
    rfe_model.fit(train_features[top_features], train_targets)

    # Sort by |coef| (biggest first), then round, then reset index
    log_reg_importance = pd.DataFrame({
        "Feature": list(top_features),
        "Importance": rfe_model.coef_[0],
    })
    log_reg_importance = log_reg_importance.sort_values(
        by="Importance", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)
    log_reg_importance["Importance"] = log_reg_importance["Importance"].round(4)

    log_reg_metrics = ModelMetrics("Logistic Regression", train_metrics, test_metrics, log_reg_importance)
    return log_reg_metrics, model


def calculate_decision_tree_metrics( train_features: pd.DataFrame, test_features: pd.DataFrame, train_targets: pd.Series, test_targets: pd.Series, n_feat_importance:int, dt_kwargs) -> tuple[ModelMetrics, DecisionTreeClassifier]:
    # TODO: Read the function description in
    # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task4.html
    # and implement the function as described.

    model = DecisionTreeClassifier(**dt_kwargs)
    model.fit(train_features, train_targets)

    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)

    train_probabilities = model.predict_proba(train_features)[:, 1]
    test_probabilities = model.predict_proba(test_features)[:, 1]

    train_tn, train_fp, train_fn, train_tp = confusion_matrix(
        train_targets,
        train_predictions
    ).ravel()

    train_fpr = train_fp / (train_fp + train_tn)
    train_fnr = train_fn / (train_fn + train_tp)

    test_tn, test_fp, test_fn, test_tp = confusion_matrix(
        test_targets,
        test_predictions
    ).ravel()

    test_fpr = test_fp / (test_fp + test_tn)
    test_fnr = test_fn / (test_fn + test_tp)
    train_metrics = {
        "accuracy": round(
            accuracy_score(train_targets, train_predictions), 4
        ),
        "recall": round(
            recall_score(train_targets, train_predictions), 4
        ),
        "precision": round(
            precision_score(train_targets, train_predictions), 4
        ),
        "fscore": round(
            f1_score(train_targets, train_predictions), 4
        ),
        "fpr": round(train_fpr, 4),
        "fnr": round(train_fnr, 4),
        "roc_auc": round(
            roc_auc_score(train_targets, train_probabilities), 4
        )
    }

    # Testing metrics.
    test_metrics = {
        "accuracy": round(
            accuracy_score(test_targets, test_predictions), 4
        ),
        "recall": round(
            recall_score(test_targets, test_predictions), 4
        ),
        "precision": round(
            precision_score(test_targets, test_predictions), 4
        ),
        "fscore": round(
            f1_score(test_targets, test_predictions), 4
        ),
        "fpr": round(test_fpr, 4),
        "fnr": round(test_fnr, 4),
        "roc_auc": round(
            roc_auc_score(test_targets, test_probabilities), 4
        )
    }

    # Pair each feature with its built-in decision-tree importance.
    dt_importance = pd.DataFrame({
        "Feature": train_features.columns,
        "Importance": model.feature_importances_
    })

    # Sort using the unrounded values.
    dt_importance = dt_importance.sort_values(
        by="Importance",
        ascending=False
    )

    # Keep only the top N features.
    dt_importance = dt_importance.head(n_feat_importance)

    # Round after sorting and selecting.
    dt_importance["Importance"] = (
        dt_importance["Importance"].round(4)
    )

    # Reset index to 0 through N-1.
    dt_importance = dt_importance.reset_index(drop=True)

    dt_metrics = ModelMetrics(
        "Decision Tree",
        train_metrics,
        test_metrics,
        dt_importance
    )

    return dt_metrics, model


def calculate_gradient_boosting_metrics(train_features:pd.DataFrame, test_features:pd.DataFrame, train_targets:pd.Series, test_targets:pd.Series, n_feat_importance: int, gb_kwargs) -> tuple[ModelMetrics,GradientBoostingClassifier]:
    # TODO: Read the function description in
    # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task4.html
    # and implement the function as described.

    model = GradientBoostingClassifier(**gb_kwargs)
    model.fit(train_features, train_targets)

    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)

    train_probabilities = model.predict_proba(train_features)[:, 1]
    test_probabilities = model.predict_proba(test_features)[:, 1]

    train_tn, train_fp, train_fn, train_tp = confusion_matrix(
        train_targets,
        train_predictions
    ).ravel()

    train_fpr = train_fp / (train_fp + train_tn)
    train_fnr = train_fn / (train_fn + train_tp)

    test_tn, test_fp, test_fn, test_tp = confusion_matrix(
        test_targets,
        test_predictions
    ).ravel()

    test_fpr = test_fp / (test_fp + test_tn)
    test_fnr = test_fn / (test_fn + test_tp)

    train_metrics = {
        "accuracy": round(
            accuracy_score(train_targets, train_predictions), 4
        ),
        "recall": round(
            recall_score(train_targets, train_predictions), 4
        ),
        "precision": round(
            precision_score(train_targets, train_predictions), 4
        ),
        "fscore": round(
            f1_score(train_targets, train_predictions), 4
        ),
        "fpr": round(train_fpr, 4),
        "fnr": round(train_fnr, 4),
        "roc_auc": round(
            roc_auc_score(train_targets, train_probabilities), 4
        )
    }

    # Testing metrics.
    test_metrics = {
        "accuracy": round(
            accuracy_score(test_targets, test_predictions), 4
        ),
        "recall": round(
            recall_score(test_targets, test_predictions), 4
        ),
        "precision": round(
            precision_score(test_targets, test_predictions), 4
        ),
        "fscore": round(
            f1_score(test_targets, test_predictions), 4
        ),
        "fpr": round(test_fpr, 4),
        "fnr": round(test_fnr, 4),
        "roc_auc": round(
            roc_auc_score(test_targets, test_probabilities), 4
        )
    }

    gb_importance = pd.DataFrame({
        "Feature": train_features.columns,
        "Importance":model.feature_importances_
    })

    gb_importance = gb_importance.sort_values(
        by="Importance",
        ascending=False
)
    gb_importance = gb_importance.head(n_feat_importance)

    gb_importance["Importance"] = (
    gb_importance["Importance"].round(4)
)

    gb_importance = gb_importance.reset_index(drop=True)


    gb_metrics = ModelMetrics(
        "Decision Tree",
        train_metrics,
        test_metrics,
        gb_importance
    )

    return gb_metrics,model

