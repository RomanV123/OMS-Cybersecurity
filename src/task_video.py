import sys
import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance

# for the main section at the end

# normal imports for this


# ************ Model Metrics
# This section is not to be changed by the student
# This packages the results and hands them to the autograder or local tests
class ModelMetrics:
    def __init__(self, model_type: str, train_metrics: dict, test_metrics: dict, feature_importance_df: pd.DataFrame):
        self.model_type = model_type
        self.train_metrics = train_metrics
        self.test_metrics = test_metrics
        self.feat_imp_df = feature_importance_df
        self.feat_name_col = "Feature"
        self.imp_col = "Importance"

    def add_train_metric(self, metric_name: str, metric_val: float):
        self.train_metrics[metric_name] = metric_val

    def add_test_metric(self, metric_name: str, metric_val: float):
        self.test_metrics[metric_name] = metric_val

    def __str__(self):
        output_str = f"MODEL TYPE: {self.model_type}\n"
        output_str += f"TRAINING METRICS:\n"
        for key in sorted(self.train_metrics.keys()):
            output_str += f"  - {key} : {self.train_metrics[key]:.4f}\n"
        output_str += f"TESTING METRICS:\n"
        for key in sorted(self.test_metrics.keys()):
            output_str += f"  - {key} : {self.test_metrics[key]:.4f}\n"
        if self.feat_imp_df is not None:
            output_str += f"FEATURE IMPORTANCES:\n"
            for i in self.feat_imp_df.index:
                output_str += f"  - {self.feat_imp_df[self.feat_name_col][i]} : {self.feat_imp_df[self.imp_col][i]:.4f}\n"
        return output_str
# ***** DON'T CHANGE THIS CLASS ABOVE

class NaiveBayes:
    def __init__(self):
        pass

    def calculate_naive_bayes_metrics(self) -> ModelMetrics:

        gnb = GaussianNB()

        # do some things with y_pred compared to the known data
        # you'll call functions from sklearn to get these values based on the data
        # we're providing the correct values here from an actual run
        accuracy = .9756
        recall = .9865
        precision = .9655
        fscore = .9759

        train_metrics = {
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "fscore": fscore
        }

        #do some things with y_pred compared to the known data
        # you'll call functions from sklearn to get these values based on the data
        # we're providing the correct values here from an actual run
        accuracy = 0.9726
        recall = 0.9861
        precision = .9603
        fscore = .973

        test_metrics = {
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "fscore": fscore
        }

        # Do something to calculate the importance
        import_mean = pd.DataFrame()

        return ModelMetrics("Naive Bayes", train_metrics, test_metrics, import_mean)
