import unittest
import pandas as pd
import os
import sys
import pickle
from src.task3 import *
from tests.utils import *

if os.path.split(os.getcwd())[-1]=="Student_Local_Testing":
    folder_loc="main"
    sys.path.append(os.getcwd())
elif os.path.split(os.getcwd())[-1]=="tests":
    folder_loc="tests"
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"..")))
else:
    raise Exception(f"Running Tests from `{os.path.split(os.getcwd())[-1]}`. Please run the tests with your CWD set to either Student_Local_Testing or tests folders")


class TestKMeansClustering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if folder_loc=="main":
            folder_path3 = os.path.join(os.getcwd(),"task3")
        else:
            folder_path3 = os.path.join(os.getcwd(),"..","task3")
        pkl_files_folder = os.path.join(folder_path3,"pkl_files")
        # Inputs
        cls.kmeans_train_dataset = pd.read_csv(os.path.join(folder_path3, "NATICUSdroid_Train_Features.csv"), index_col=0)
        cls.kmeans_test_dataset = pd.read_csv(os.path.join(folder_path3, "NATICUSdroid_Test_Features.csv"), index_col=0)

        # Answers
        try:
            cls.kmeans_train_cluster_ids_w_k = pd.read_pickle(os.path.join(pkl_files_folder, "kmc_train_ids_w_k.pkl"))
            cls.kmeans_train_cluster_ids = pd.read_pickle(os.path.join(pkl_files_folder, "kmc_train_ids.pkl"))
            cls.kmeans_test_cluster_ids = pd.read_pickle(os.path.join(pkl_files_folder, "kmc_test_ids.pkl"))
        except:
            print("Missing answer pkl files")

        cls.optimal_k = 8
        cls.k = 4
        random_state = 0
        algorithm='elkan'
        init='k-means++'
        n_init=1
        max_iter=300
        tol=1e-8
        cls.kmc = KmeansClustering(random_state=random_state, init=init, n_init=n_init, max_iter=max_iter, algorithm=algorithm,tol=tol)
    def test_kmeans_get_n_clusters(self):
        self.kmc.optimal_k = self.kmc.kmeans_get_n_clusters(train_features=self.kmeans_train_dataset)
        self.assertTrue(compare_submission_to_answer(self.kmc.optimal_k,self.optimal_k,"Kmeans Determine Optimal K"))
    
    def test_kmeans_train_no_k(self):
        self.kmc.train_ids = self.kmc.kmeans_train(train_features=self.kmeans_train_dataset)
        self.assertTrue(compare_submission_to_answer(self.kmc.train_ids,self.kmeans_train_cluster_ids,"Kmeans Train Cluster IDs without k"))
    
    def test_kmeans_train_with_k(self):
        self.kmc.train_ids = self.kmc.kmeans_train(train_features=self.kmeans_train_dataset,k=self.k)
        self.assertTrue(compare_submission_to_answer(self.kmc.train_ids,self.kmeans_train_cluster_ids_w_k,"Kmeans Train Cluster IDs with k provided"))
    
    def test_kmeans_test(self):
        _ = self.kmc.kmeans_train(self.kmeans_train_dataset)
        self.kmc.test_ids = self.kmc.kmeans_test(self.kmeans_test_dataset)
        self.assertTrue(compare_submission_to_answer(self.kmc.test_ids,self.kmeans_test_cluster_ids,"Kmeans Test Cluster IDs"))


if __name__ == '__main__':
    unittest.main()

