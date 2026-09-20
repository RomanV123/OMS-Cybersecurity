import unittest
import pandas as pd
import os
import sys
import pickle
from src.task2 import *
from tests.utils import *
import ipaddress

if os.path.split(os.getcwd())[-1]=="Student_Local_Testing":
    folder_loc="main"
    sys.path.append(os.getcwd())
elif os.path.split(os.getcwd())[-1]=="tests":
    folder_loc="tests"
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"..")))
else:
    raise Exception(f"Running Tests from `{os.path.split(os.getcwd())[-1]}`. Please run the tests with your CWD set to either Student_Local_Testing or tests folders")


class TestTrainTestSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if folder_loc=="main":
            folder_path = os.path.join(os.getcwd(),"task2")
        else:
            folder_path = os.path.join(os.getcwd(),"..","task2")
        cls.dataset = pd.read_csv(os.path.join(folder_path,"sample.csv"), index_col=0)
        pkl_files_folder = os.path.join(folder_path,"pkl_files")
        try:
            cls.ans_train_features = pd.read_pickle(os.path.join(pkl_files_folder,"train_feats_tts.pkl"))
            cls.ans_test_features = pd.read_pickle(os.path.join(pkl_files_folder,"test_feats_tts.pkl"))
            cls.ans_train_targets = pd.read_pickle(os.path.join(pkl_files_folder,"train_targets_tts.pkl"))
            cls.ans_test_targets = pd.read_pickle(os.path.join(pkl_files_folder,"test_targets_tts.pkl"))
        except:
            print("TestTrainTestSplit Missing answer pkl files")
        cls.target_col = "flagged"
        cls.train_features,cls.test_features,cls.train_targets,cls.test_targets = tts(cls.dataset,
                            cls.target_col, 
                            test_size=.2,
                            should_stratify=True,
                            random_state=0)
    def test_train_features(self):
        self.assertTrue(compare_submission_to_answer_df(self.train_features,self.ans_train_features,"Train Features DF"))
            
    def test_test_features(self):
        self.assertTrue(compare_submission_to_answer_df(self.test_features,self.ans_test_features,"Test Features DF"))
    
    def test_train_targets(self):
        self.assertTrue(compare_submission_to_answer_series(self.train_targets,self.ans_train_targets,"Train Targets Series"))
    
    def test_test_targets(self):
        self.assertTrue(compare_submission_to_answer_series(self.test_targets,self.ans_test_targets,"Test Targets Series"))

def get_hour_of_day(dataframe:pd.DataFrame):
    return pd.to_datetime(dataframe["time"]).dt.hour

class TestOneHotEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if folder_loc=="main":
            folder_path2 = os.path.join(os.getcwd(),"task2")
        else:
            folder_path2 = os.path.join(os.getcwd(),"..","task2")
        pkl_files_folder = os.path.join(folder_path2,"pkl_files")
        try:
            # Inputs
            cls.ans_train_features = pd.read_pickle(os.path.join(pkl_files_folder,"train_feats_tts.pkl"))
            cls.ans_test_features = pd.read_pickle(os.path.join(pkl_files_folder,"test_feats_tts.pkl"))
            cls.ans_train_targets = pd.read_pickle(os.path.join(pkl_files_folder,"train_targets_tts.pkl"))
            cls.ans_test_targets = pd.read_pickle(os.path.join(pkl_files_folder,"test_targets_tts.pkl"))
        except:
            print("TestOneHotEncoder Missing input pkl files")
        try:
            # Answers
            cls.ans_train_features_ohe = pd.read_pickle(os.path.join(pkl_files_folder,"train_feats_ohe.pkl"))
            cls.ans_test_features_ohe = pd.read_pickle(os.path.join(pkl_files_folder,"test_feats_ohe.pkl"))
        except:
            print("TestOneHotEncoder Missing answer pkl files")
        cls.target_col = "flagged"
        cls.preprocessDataset = PreprocessDataset(
                       one_hot_encode_cols = ["src_ip","protocol"],
                       min_max_scale_cols = ["bytes_in","bytes_out"],
                       n_components = 2,
                       feature_engineering_functions = {"time_hour":get_hour_of_day})
    def test_train_features_ohe(self):
        train_features_ohe = self.preprocessDataset.one_hot_encode_columns_train(train_features = self.ans_train_features)
        self.assertTrue(compare_submission_to_answer_df(train_features_ohe,self.ans_train_features_ohe,"One Hot Encoded Train DF"))
    
    def test_test_features_ohe(self):
        _ = self.preprocessDataset.one_hot_encode_columns_train(train_features = self.ans_train_features)
        test_features_ohe = self.preprocessDataset.one_hot_encode_columns_test(test_features = self.ans_test_features)
        self.assertTrue(compare_submission_to_answer_df(test_features_ohe,self.ans_test_features_ohe,"One Hot Encoded Test DF"))

class TestMinMaxScaler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if folder_loc=="main":
            folder_path2 = os.path.join(os.getcwd(),"task2")
        else:
            folder_path2 = os.path.join(os.getcwd(),"..","task2")
        pkl_files_folder = os.path.join(folder_path2,"pkl_files")
        try:
            # Inputs
            cls.ans_train_features = pd.read_pickle(os.path.join(pkl_files_folder,"train_feats_tts.pkl"))
            cls.ans_test_features = pd.read_pickle(os.path.join(pkl_files_folder,"test_feats_tts.pkl"))
            cls.ans_train_targets = pd.read_pickle(os.path.join(pkl_files_folder,"train_targets_tts.pkl"))
            cls.ans_test_targets = pd.read_pickle(os.path.join(pkl_files_folder,"test_targets_tts.pkl"))
        except:
            print("TestMinMaxScaler Missing input pkl files")
        try:
            # Answers
            cls.ans_train_features_mms = pd.read_pickle(os.path.join(pkl_files_folder,"train_feats_mms.pkl"))
            cls.ans_test_features_mms = pd.read_pickle(os.path.join(pkl_files_folder,"test_feats_mms.pkl"))
        except:
            print("TestMinMaxScaler Missing answer pkl files")
        cls.target_col = "flagged"
        cls.preprocessDataset = PreprocessDataset(
                       one_hot_encode_cols = ["src_ip","protocol"],
                       min_max_scale_cols = ["bytes_in","bytes_out"],
                       n_components = 2,
                       feature_engineering_functions = {"time_hour":get_hour_of_day})
    def test_train_features_mms(self):
        train_features_mms = self.preprocessDataset.min_max_scaled_columns_train(train_features = self.ans_train_features)
        self.assertTrue(compare_submission_to_answer_df(train_features_mms,self.ans_train_features_mms,"Min Max Scaled Train DF"))
    
    def test_test_features_mms(self):
        _ = self.preprocessDataset.min_max_scaled_columns_train(train_features = self.ans_train_features)
        test_features_mms = self.preprocessDataset.min_max_scaled_columns_test(test_features = self.ans_test_features)
        self.assertTrue(compare_submission_to_answer_df(test_features_mms,self.ans_test_features_mms,"Min Max Scaled Test DF"))

class TestPCA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if folder_loc=="main":
            folder_path2 = os.path.join(os.getcwd(),"task2")
        else:
            folder_path2 = os.path.join(os.getcwd(),"..","task2")
        pkl_files_folder = os.path.join(folder_path2,"pkl_files")
        try:
            # Inputs
            cls.train_feats_pca_inputs = pd.read_pickle(os.path.join(pkl_files_folder,"train_feats_pca_inputs.pkl"))
            cls.test_feats_pca_inputs = pd.read_pickle(os.path.join(pkl_files_folder,"test_feats_pca_inputs.pkl"))
        except:
            print("TestPCA Missing input pkl files")
        try:
            # Answers
            cls.ans_train_features_pca = pd.read_pickle(os.path.join(pkl_files_folder,"train_feats_pca.pkl"))
            cls.ans_test_features_pca = pd.read_pickle(os.path.join(pkl_files_folder,"test_feats_pca.pkl"))
        except:
            print("TestPCA Missing answer pkl files")
        cls.target_col = "flagged"
        cls.preprocessDataset = PreprocessDataset(
                       one_hot_encode_cols = [],
                       min_max_scale_cols = ["bytes_in","bytes_out","src_ip","time"],
                       n_components = 2,
                       feature_engineering_functions = {"time_hour":get_hour_of_day})
    def test_train_features_pca(self):
        train_features_pca = self.preprocessDataset.pca_train(train_features = self.train_feats_pca_inputs)
        self.assertTrue(compare_submission_to_answer_df(train_features_pca.round(4),self.ans_train_features_pca.round(4),"PCA Train DF"))
    
    
    def test_test_features_pca(self):
        _ = self.preprocessDataset.pca_train(train_features = self.train_feats_pca_inputs)
        test_features_pca = self.preprocessDataset.pca_test(test_features = self.test_feats_pca_inputs)
        self.assertTrue(compare_submission_to_answer_df(test_features_pca.round(4),self.ans_test_features_pca.round(4),"PCA Test DF"))

class TestFeatureEngineering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if folder_loc=="main":
            folder_path2 = os.path.join(os.getcwd(),"task2")
        else:
            folder_path2 = os.path.join(os.getcwd(),"..","task2")
        pkl_files_folder = os.path.join(folder_path2,"pkl_files")
        try:
            # Inputs
            cls.ans_train_features = pd.read_pickle(os.path.join(pkl_files_folder,"train_feats_tts.pkl"))
            cls.ans_test_features = pd.read_pickle(os.path.join(pkl_files_folder,"test_feats_tts.pkl"))
            cls.ans_train_targets = pd.read_pickle(os.path.join(pkl_files_folder,"train_targets_tts.pkl"))
            cls.ans_test_targets = pd.read_pickle(os.path.join(pkl_files_folder,"test_targets_tts.pkl"))
        except:
            print("TestFeatureEngineering Missing input pkl files")
        try:
            # Answers
            cls.ans_train_features_fe = pd.read_pickle(os.path.join(pkl_files_folder,"train_feats_fe.pkl"))
            cls.ans_test_features_fe = pd.read_pickle(os.path.join(pkl_files_folder,"test_feats_fe.pkl"))
        except:
            print("TestFeatureEngineering Missing answer pkl files")
        cls.target_col = "flagged"
        cls.preprocessDataset = PreprocessDataset(
                       one_hot_encode_cols = ["src_ip","protocol"],
                       min_max_scale_cols = ["bytes_in","bytes_out"],
                       n_components = 2,
                       feature_engineering_functions = {"time_hour":get_hour_of_day})
    def test_train_features_fe(self):
        train_features_fe = self.preprocessDataset.feature_engineering_train(train_features = self.ans_train_features)
        self.assertTrue(compare_submission_to_answer_df(train_features_fe,self.ans_train_features_fe,"Feature Engineered Train DF"))
    
    def test_test_features_fe(self):
        test_features_fe = self.preprocessDataset.feature_engineering_test(test_features = self.ans_test_features)
        self.assertTrue(compare_submission_to_answer_df(test_features_fe,self.ans_test_features_fe,"Feature Engineered Test DF"))

class TestPreprocess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if folder_loc=="main":
            folder_path2 = os.path.join(os.getcwd(),"task2")
        else:
            folder_path2 = os.path.join(os.getcwd(),"..","task2")
        pkl_files_folder = os.path.join(folder_path2,"pkl_files")
        try:
            # Inputs
            cls.ans_train_features = pd.read_pickle(os.path.join(pkl_files_folder,"train_feats_tts.pkl"))
            cls.ans_test_features = pd.read_pickle(os.path.join(pkl_files_folder,"test_feats_tts.pkl"))
            cls.ans_train_targets = pd.read_pickle(os.path.join(pkl_files_folder,"train_targets_tts.pkl"))
            cls.ans_test_targets = pd.read_pickle(os.path.join(pkl_files_folder,"test_targets_tts.pkl"))
        except:
            print("TestPreprocess Missing input pkl files")
        try:
            # Answers
            cls.ans_train_features_preprocess = pd.read_pickle(os.path.join(pkl_files_folder,"train_feats_preprocess.pkl"))
            cls.ans_test_features_preprocess = pd.read_pickle(os.path.join(pkl_files_folder,"test_feats_preprocess.pkl"))
        except:
            print("TestPreprocess Missing answer pkl files")
        cls.target_col = "flagged"
        cls.preprocessDataset = PreprocessDataset(
                       one_hot_encode_cols = ["src_ip","protocol"],
                       min_max_scale_cols = ["bytes_in","bytes_out"],
                       n_components = 2,
                       feature_engineering_functions = {"time_hour":get_hour_of_day})
    def test_train_features_preprocess(self):
        self.train_features_preprocess = self.preprocessDataset.preprocess_train(train_features = self.ans_train_features)
        self.assertTrue(compare_submission_to_answer_df(self.train_features_preprocess,self.ans_train_features_preprocess,"Preprocessed Train DF"))
    
    def test_test_features_preprocess(self):
        _ = self.preprocessDataset.preprocess_train(train_features = self.ans_train_features)
        self.test_features_preprocess = self.preprocessDataset.preprocess_test(test_features = self.ans_test_features)
        self.assertTrue(compare_submission_to_answer_df(self.test_features_preprocess,self.ans_test_features_preprocess,"Preprocessed Test DF"))


if __name__ == '__main__':
    unittest.main()

