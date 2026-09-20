import unittest
import pandas as pd
import os
import sys
import pickle
from src.task4 import *
from tests.utils import *

if os.path.split(os.getcwd())[-1]=="Student_Local_Testing":
    folder_loc="main"
    sys.path.append(os.getcwd())
elif os.path.split(os.getcwd())[-1]=="tests":
    folder_loc="tests"
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"..")))
else:
    raise Exception(f"Running Tests from `{os.path.split(os.getcwd())[-1]}`. Please run the tests with your CWD set to either Student_Local_Testing or tests folders")
class Test_calculate_naive_metrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Train and Test Data is derived from https://archive.ics.uci.edu/dataset/722/naticusdroid+android+permissions+dataset
        if folder_loc=="main":
            folder_path4 = os.path.join(os.getcwd(),"task4")
        else:
            folder_path4 = os.path.join(os.getcwd(),"..","task4")
        pkl_files_folder = os.path.join(folder_path4,"pkl_files")
        # Inputs
        cls.train_features = pd.read_csv(os.path.join(folder_path4,"NATICUSdroid_Train_Features.csv"), index_col=0)
        cls.test_features  = pd.read_csv(os.path.join(folder_path4,"NATICUSdroid_Test_Features.csv"), index_col=0)
        cls.train_targets  = pd.read_csv(os.path.join(folder_path4,"NATICUSdroid_Train_Targets.csv"), index_col=0)["Result"]
        cls.test_targets   = pd.read_csv(os.path.join(folder_path4,"NATICUSdroid_Test_Targets.csv"), index_col=0)["Result"]
        
        cls.naive_assumption = 1

        # Answers for NATICUSdroid Dataset
        try:
            with open(os.path.join(pkl_files_folder,"naive_metrics.pkl"), 'rb') as file:
                cls.naive_metrics_ans = pickle.load(file)
        except:
            print("Missing answer pkl files")
            
        # Calculate Metrics with Student's Function
        cls.naive_metrics = calculate_naive_metrics(cls.train_features, cls.test_features, cls.train_targets, cls.test_targets, cls.naive_assumption)
    def test_train_accuracy(self):
        train_accuracy = self.naive_metrics.train_metrics["accuracy"]
        train_accuracy_ans = self.naive_metrics_ans.train_metrics["accuracy"]
        self.assertEqual(train_accuracy, train_accuracy_ans)
    
    def test_train_recall(self):
        train_recall = self.naive_metrics.train_metrics["recall"]
        train_recall_ans = self.naive_metrics_ans.train_metrics["recall"]
        self.assertEqual(train_recall_ans, train_recall)
    
    def test_train_precision(self):
        train_precision = self.naive_metrics.train_metrics["precision"]
        train_precision_ans = self.naive_metrics_ans.train_metrics["precision"]
        self.assertEqual(train_precision_ans, train_precision)
    
    def test_train_fscore(self):
        train_fscore = self.naive_metrics.train_metrics["fscore"]
        train_fscore_ans = self.naive_metrics_ans.train_metrics["fscore"]
        self.assertEqual(train_fscore_ans, train_fscore)
    
    def test_test_accuracy(self):
        test_accuracy = self.naive_metrics.test_metrics["accuracy"]
        test_accuracy_ans = self.naive_metrics_ans.test_metrics["accuracy"]
        self.assertEqual(test_accuracy_ans, test_accuracy)
    
    def test_test_recall(self):
        train_recall = self.naive_metrics.test_metrics["recall"]
        test_recall_ans = self.naive_metrics_ans.test_metrics["recall"]
        self.assertEqual(test_recall_ans, train_recall)
    
    def test_test_precision(self):
        train_precision = self.naive_metrics.test_metrics["precision"]
        test_precision_ans = self.naive_metrics_ans.test_metrics["precision"]
        self.assertEqual(test_precision_ans, train_precision)
    
    def test_test_fscore(self):
        train_fscore = self.naive_metrics.test_metrics["fscore"]
        test_fscore_ans = self.naive_metrics_ans.test_metrics["fscore"]
        self.assertEqual(test_fscore_ans, train_fscore)
class Test_calculate_logistic_regression_metrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Train and Test Data is derived from https://archive.ics.uci.edu/dataset/722/naticusdroid+android+permissions+dataset
        if folder_loc=="main":
            folder_path4 = os.path.join(os.getcwd(),"task4")
        else:
            folder_path4 = os.path.join(os.getcwd(),"..","task4")
        pkl_files_folder = os.path.join(folder_path4,"pkl_files")
        # Inputs
        cls.train_features = pd.read_csv(os.path.join(folder_path4,"NATICUSdroid_Train_Features.csv"), index_col=0)
        cls.test_features  = pd.read_csv(os.path.join(folder_path4,"NATICUSdroid_Test_Features.csv"), index_col=0)
        cls.train_targets  = pd.read_csv(os.path.join(folder_path4,"NATICUSdroid_Train_Targets.csv"), index_col=0)["Result"]
        cls.test_targets   = pd.read_csv(os.path.join(folder_path4,"NATICUSdroid_Test_Targets.csv"), index_col=0)["Result"]

        # Answers for NATICUSdroid Dataset
        try:
            with open(os.path.join(pkl_files_folder,"logreg_metrics.pkl"), 'rb') as file:
                cls.logreg_metrics_ans = pickle.load(file)
            cls.importance_df_ans = pd.read_pickle(os.path.join(pkl_files_folder,"logreg_importance.pkl"))
        except:
            print("Missing answer pkl files")
            
        cls.logreg_kwargs = {'penalty':'l1','fit_intercept':False,'solver':'liblinear','random_state':0}
        cls.n_feat_importance = 10
        # Calculate Metrics with Student's Function
        cls.logreg_metrics,_ = calculate_logistic_regression_metrics(cls.train_features, cls.test_features, cls.train_targets, cls.test_targets,cls.n_feat_importance, cls.logreg_kwargs)
    def test_train_accuracy(self):
        train_accuracy = self.logreg_metrics.train_metrics["accuracy"]
        train_accuracy_ans = self.logreg_metrics_ans.train_metrics["accuracy"]
        self.assertEqual(train_accuracy_ans,train_accuracy)
    
    def test_train_recall(self):
        train_recall = self.logreg_metrics.train_metrics["recall"]
        train_recall_ans = self.logreg_metrics_ans.train_metrics["recall"]
        self.assertEqual(train_recall_ans,train_recall)
    
    def test_train_precision(self):
        train_precision = self.logreg_metrics.train_metrics["precision"]
        train_precision_ans = self.logreg_metrics_ans.train_metrics["precision"]
        self.assertEqual(train_precision_ans,train_precision)
    
    def test_train_fscore(self):
        train_fscore = self.logreg_metrics.train_metrics["fscore"]
        train_fscore_ans = self.logreg_metrics_ans.train_metrics["fscore"]
        self.assertEqual(train_fscore_ans,train_fscore)
    
    def test_train_fpr(self):
        train_fpr = self.logreg_metrics.train_metrics["fpr"]
        train_fpr_ans = self.logreg_metrics_ans.train_metrics["fpr"]
        self.assertEqual(train_fpr_ans,train_fpr)
    
    def test_train_fnr(self):
        train_fnr = self.logreg_metrics.train_metrics["fnr"]
        train_fnr_ans = self.logreg_metrics_ans.train_metrics["fnr"]
        self.assertEqual(train_fnr_ans,train_fnr)
        
    def test_train_roc_auc(self):
        train_roc_auc = self.logreg_metrics.train_metrics["roc_auc"]
        train_roc_auc_ans = self.logreg_metrics_ans.train_metrics["roc_auc"]
        self.assertEqual(train_roc_auc_ans,train_roc_auc)
    
    def test_test_accuracy(self):
        test_accuracy = self.logreg_metrics.test_metrics["accuracy"]
        test_accuracy_ans = self.logreg_metrics_ans.test_metrics["accuracy"]
        self.assertEqual(test_accuracy_ans,test_accuracy)
    
    def test_test_recall(self):
        test_recall = self.logreg_metrics.test_metrics["recall"]
        test_recall_ans = self.logreg_metrics_ans.test_metrics["recall"]
        self.assertEqual(test_recall_ans, test_recall)
    
    def test_test_precision(self):
        test_precision = self.logreg_metrics.test_metrics["precision"]
        test_precision_ans = self.logreg_metrics_ans.test_metrics["precision"]
        self.assertEqual(test_precision_ans, test_precision)
    
    def test_test_fscore(self):
        test_fscore = self.logreg_metrics.test_metrics["fscore"]
        test_fscore_ans = self.logreg_metrics_ans.test_metrics["fscore"]
        self.assertEqual(test_fscore_ans, test_fscore)
    
    def test_test_fpr(self):
        test_fpr = self.logreg_metrics.test_metrics["fpr"]
        test_fpr_ans = self.logreg_metrics_ans.test_metrics["fpr"]
        self.assertEqual(test_fpr_ans, test_fpr)
    
    def test_test_fnr(self):
        test_fnr = self.logreg_metrics.test_metrics["fnr"]
        test_fnr_ans = self.logreg_metrics_ans.test_metrics["fnr"]
        self.assertEqual(test_fnr_ans, test_fnr)
        
    def test_test_roc_auc(self):
        test_roc_auc = self.logreg_metrics.test_metrics["roc_auc"]
        test_roc_auc_ans = self.logreg_metrics_ans.test_metrics["roc_auc"]
        self.assertEqual(test_roc_auc_ans, test_roc_auc)
    
    def test_importance(self):
        importance_df = self.logreg_metrics.feat_imp_df
        self.assertTrue(compare_submission_to_answer_df(importance_df.round(4),self.importance_df_ans,"LR Feature Importance DF"))

class Test_calculate_decision_tree_metrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Train and Test Data is derived from https://archive.ics.uci.edu/dataset/722/naticusdroid+android+permissions+dataset
        if folder_loc == "main":
            folder_path4 = os.path.join(os.getcwd(), "task4")
        else:
            folder_path4 = os.path.join(os.getcwd(), "..", "task4")
        pkl_files_folder = os.path.join(folder_path4, "pkl_files")
        # Inputs
        cls.train_features = pd.read_csv(os.path.join(folder_path4, "NATICUSdroid_Train_Features.csv"), index_col=0)
        cls.test_features = pd.read_csv(os.path.join(folder_path4, "NATICUSdroid_Test_Features.csv"), index_col=0)
        cls.train_targets = pd.read_csv(os.path.join(folder_path4, "NATICUSdroid_Train_Targets.csv"), index_col=0)["Result"]
        cls.test_targets = pd.read_csv(os.path.join(folder_path4, "NATICUSdroid_Test_Targets.csv"), index_col=0)["Result"]

        # Answers for NATICUSdroid Dataset
        try:
            with open(os.path.join(pkl_files_folder,"dt_metrics.pkl"), 'rb') as file:
                cls.dt_metrics_ans = pickle.load(file)
            cls.importance_df_ans = pd.read_pickle(os.path.join(pkl_files_folder,"dt_importance.pkl"))
        except:
            print("Missing answer pkl files")

        cls.dt_kwargs = {'criterion': 'entropy', 'max_depth': 3, 'random_state': 0}
        cls.n_feat_importance = 5
        # Calculate Metrics with Student's Function
        cls.dt_metrics, _ = calculate_decision_tree_metrics(
            cls.train_features,
            cls.test_features,
            cls.train_targets,
            cls.test_targets,
            cls.n_feat_importance,
            cls.dt_kwargs
        )
    def test_train_accuracy(self):
        self.assertEqual(self.dt_metrics_ans.train_metrics["accuracy"], self.dt_metrics.train_metrics["accuracy"])
    
    def test_train_recall(self):
        self.assertEqual(self.dt_metrics_ans.train_metrics["recall"], self.dt_metrics.train_metrics["recall"])
    
    def test_train_precision(self):
        self.assertEqual(self.dt_metrics_ans.train_metrics["precision"], self.dt_metrics.train_metrics["precision"])
    
    def test_train_fscore(self):
        self.assertEqual(self.dt_metrics_ans.train_metrics["fscore"], self.dt_metrics.train_metrics["fscore"])
    
    def test_train_fpr(self):
        self.assertEqual(self.dt_metrics_ans.train_metrics["fpr"], self.dt_metrics.train_metrics["fpr"])
    
    def test_train_fnr(self):
        self.assertEqual(self.dt_metrics_ans.train_metrics["fnr"], self.dt_metrics.train_metrics["fnr"])
        
    def test_train_roc_auc(self):
        self.assertEqual(self.dt_metrics_ans.train_metrics["roc_auc"], self.dt_metrics.train_metrics["roc_auc"])
    
    def test_test_accuracy(self):
        self.assertEqual(self.dt_metrics_ans.test_metrics["accuracy"], self.dt_metrics.test_metrics["accuracy"])
    
    def test_test_recall(self):
        self.assertEqual(self.dt_metrics_ans.test_metrics["recall"], self.dt_metrics.test_metrics["recall"])
    
    def test_test_precision(self):
        self.assertEqual(self.dt_metrics_ans.test_metrics["precision"], self.dt_metrics.test_metrics["precision"])
    
    def test_test_fscore(self):
        self.assertEqual(self.dt_metrics_ans.test_metrics["fscore"], self.dt_metrics.test_metrics["fscore"])
    
    def test_test_fpr(self):
        self.assertEqual(self.dt_metrics_ans.test_metrics["fpr"], self.dt_metrics.test_metrics["fpr"])
    
    def test_test_fnr(self):
        self.assertEqual(self.dt_metrics_ans.test_metrics["fnr"], self.dt_metrics.test_metrics["fnr"])
        
    def test_test_roc_auc(self):
        self.assertEqual(self.dt_metrics_ans.test_metrics["roc_auc"], self.dt_metrics.test_metrics["roc_auc"])
    
    def test_importance(self):
        self.assertTrue(compare_submission_to_answer_df(self.dt_metrics.feat_imp_df.round(4), self.importance_df_ans, "DT Feature Importance DF"))

class Test_calculate_gradient_boosting_metrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Train and Test Data is derived from https://archive.ics.uci.edu/dataset/722/naticusdroid+android+permissions+dataset
        if folder_loc=="main":
            folder_path4 = os.path.join(os.getcwd(),"task4")
        else:
            folder_path4 = os.path.join(os.getcwd(),"..","task4")
        pkl_files_folder = os.path.join(folder_path4,"pkl_files")
        # Inputs
        cls.train_features = pd.read_csv(os.path.join(folder_path4,"NATICUSdroid_Train_Features.csv"), index_col=0)
        cls.test_features  = pd.read_csv(os.path.join(folder_path4,"NATICUSdroid_Test_Features.csv"), index_col=0)
        cls.train_targets  = pd.read_csv(os.path.join(folder_path4,"NATICUSdroid_Train_Targets.csv"), index_col=0)["Result"]
        cls.test_targets   = pd.read_csv(os.path.join(folder_path4,"NATICUSdroid_Test_Targets.csv"), index_col=0)["Result"]

        # Answers for NATICUSdroid Dataset
        try:
            with open(os.path.join(pkl_files_folder,"gb_metrics.pkl"), 'rb') as file:
                cls.gb_metrics_ans = pickle.load(file)
            cls.importance_df_ans = pd.read_pickle(os.path.join(pkl_files_folder,"gb_importance.pkl"))
        except:
            print("Missing answer pkl files")
        
        cls.gb_kwargs = {'n_estimators':150,'criterion':'squared_error','max_depth':3,'random_state':0}
        cls.n_feat_importance = 10
        # Calculate Metrics with Student's Function
        cls.gb_metrics,_ = calculate_gradient_boosting_metrics(cls.train_features, cls.test_features, cls.train_targets, cls.test_targets, cls.n_feat_importance, cls.gb_kwargs)
    def test_train_accuracy(self):
        train_accuracy = self.gb_metrics.train_metrics["accuracy"]
        train_accuracy_ans = self.gb_metrics_ans.train_metrics["accuracy"]
        self.assertEqual(train_accuracy_ans, train_accuracy)
    
    def test_train_recall(self):
        train_recall = self.gb_metrics.train_metrics["recall"]
        train_recall_ans = self.gb_metrics_ans.train_metrics["recall"]
        self.assertEqual(train_recall_ans, train_recall)
    
    def test_train_precision(self):
        train_precision = self.gb_metrics.train_metrics["precision"]
        train_precision_ans = self.gb_metrics_ans.train_metrics["precision"]
        self.assertEqual(train_precision_ans, train_precision)
    
    def test_train_fscore(self):
        train_fscore = self.gb_metrics.train_metrics["fscore"]
        train_fscore_ans = self.gb_metrics_ans.train_metrics["fscore"]
        self.assertEqual(train_fscore_ans, train_fscore)
    
    def test_train_fpr(self):
        train_fpr = self.gb_metrics.train_metrics["fpr"]
        train_fpr_ans = self.gb_metrics_ans.train_metrics["fpr"]
        self.assertEqual(train_fpr_ans, train_fpr)
    
    def test_train_fnr(self):
        train_fnr = self.gb_metrics.train_metrics["fnr"]
        train_fnr_ans = self.gb_metrics_ans.train_metrics["fnr"]
        self.assertEqual(train_fnr_ans, train_fnr)
        
    def test_train_roc_auc(self):
        train_roc_auc = self.gb_metrics.train_metrics["roc_auc"]
        train_roc_auc_ans = self.gb_metrics_ans.train_metrics["roc_auc"]
        self.assertEqual(train_roc_auc_ans, train_roc_auc)
    
    def test_test_accuracy(self):
        test_accuracy = self.gb_metrics.test_metrics["accuracy"]
        test_accuracy_ans = self.gb_metrics_ans.test_metrics["accuracy"]
        self.assertEqual(test_accuracy_ans, test_accuracy)
    
    def test_test_recall(self):
        test_recall = self.gb_metrics.test_metrics["recall"]
        test_recall_ans = self.gb_metrics_ans.test_metrics["recall"]
        self.assertEqual(test_recall_ans, test_recall)
    
    def test_test_precision(self):
        test_precision = self.gb_metrics.test_metrics["precision"]
        test_precision_ans = self.gb_metrics_ans.test_metrics["precision"]
        self.assertEqual(test_precision_ans, test_precision)
    
    def test_test_fscore(self):
        test_fscore = self.gb_metrics.test_metrics["fscore"]
        test_fscore_ans = self.gb_metrics_ans.test_metrics["fscore"]
        self.assertEqual(test_fscore_ans, test_fscore)
    
    def test_test_fpr(self):
        test_fpr = self.gb_metrics.test_metrics["fpr"]
        test_fpr_ans = self.gb_metrics_ans.test_metrics["fpr"]
        self.assertEqual(test_fpr_ans, test_fpr)
    
    def test_test_fnr(self):
        test_fnr = self.gb_metrics.test_metrics["fnr"]
        test_fnr_ans = self.gb_metrics_ans.test_metrics["fnr"]
        self.assertEqual(test_fnr_ans, test_fnr)
        
    def test_test_roc_auc(self):
        test_roc_auc = self.gb_metrics.test_metrics["roc_auc"]
        test_roc_auc_ans = self.gb_metrics_ans.test_metrics["roc_auc"]
        self.assertEqual(test_roc_auc_ans,test_roc_auc)
    
    def test_importance(self):
        importance_df = self.gb_metrics.feat_imp_df
        self.assertTrue(compare_submission_to_answer_df(importance_df.round(4),self.importance_df_ans,"GB Feature Importance DF"))

if __name__ == '__main__':
    unittest.main()

