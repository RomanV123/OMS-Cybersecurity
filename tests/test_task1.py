import unittest
import pandas as pd
import os
import sys
import pickle
from src.task1 import *
from tests.utils import *

if os.path.split(os.getcwd())[-1]=="Student_Local_Testing":
    folder_loc="main"
    sys.path.append(os.getcwd())
elif os.path.split(os.getcwd())[-1]=="tests":
    folder_loc="tests"
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"..")))
else:
    raise Exception(f"Running Tests from `{os.path.split(os.getcwd())[-1]}`. Please run the tests with your CWD set to either Student_Local_Testing or tests folders")


class Test_find_data_type(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if folder_loc=="main":
            folder_path = os.path.join(os.getcwd(),"task1")
        else:
            folder_path = os.path.join(os.getcwd(),"..","task1")
        try:
            cls.dataset = pd.read_csv(os.path.join(folder_path,"sample.csv"),index_col=0)
            cls.dataset["time"] = pd.to_datetime(cls.dataset["time"])
        except:
            print("Test_find_data_type Missing input files")
    def test_find_data_type_src_ip(self):
        self.assertEqual(find_data_type(self.dataset, "src_ip"), object)
    
    def test_find_data_type_protocol(self):
        self.assertEqual(find_data_type(self.dataset, "protocol"), object)
    
    def test_find_data_type_bytes_in(self):
        self.assertEqual(find_data_type(self.dataset, "bytes_in"), np.int64)
    
    def test_find_data_type_bytes_out(self):
        self.assertEqual(find_data_type(self.dataset, "bytes_out"), np.int64)
    
    def test_find_data_type_time(self):
        self.assertEqual(find_data_type(self.dataset, "time"), np.dtype('datetime64[ns]'))
    
    def test_find_data_type_flagged(self):
        self.assertEqual(find_data_type(self.dataset, "flagged"), np.int64)

class Test_set_index_col(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if folder_loc=="main":
            folder_path = os.path.join(os.getcwd(),"task1")
        else:
            folder_path = os.path.join(os.getcwd(),"..","task1")
        try:
            cls.dataset = pd.read_csv(os.path.join(folder_path,"sample.csv"),index_col=0)
        except:
            print("Test_set_index_col Missing input files")
        try:
            pkl_files_folder = os.path.join(folder_path,"pkl_files")
            cls.set_index_col_src_ip_ans = pd.read_pickle(os.path.join(pkl_files_folder,"set_index_col_src_ip.pkl"))
            cls.set_index_col_protocol_ans = pd.read_pickle(os.path.join(pkl_files_folder,"set_index_col_protocol.pkl"))
            cls.set_index_col_bytes_in_ans = pd.read_pickle(os.path.join(pkl_files_folder,"set_index_col_bytes_in.pkl"))
            cls.set_index_col_bytes_out_ans = pd.read_pickle(os.path.join(pkl_files_folder,"set_index_col_bytes_out.pkl"))
            cls.set_index_col_time_ans = pd.read_pickle(os.path.join(pkl_files_folder,"set_index_col_time.pkl"))
            cls.set_index_col_flagged_ans = pd.read_pickle(os.path.join(pkl_files_folder,"set_index_col_flagged.pkl"))
        except:
            print("Test_set_index_col Missing answer files")
    def test_set_index_col_src_ip(self):
        self.assertTrue(compare_submission_to_answer(set_index_col(self.dataset, self.dataset["src_ip"]).index.to_list(), self.dataset["src_ip"].to_list(),"set_index_col (src_ip)"))
    
    def test_set_index_col_protocol(self):
        self.assertTrue(compare_submission_to_answer(set_index_col(self.dataset, self.dataset["protocol"]).index.to_list(), self.dataset["protocol"].to_list(),"set_index_col (protocol)"))
    
    def test_set_index_col_bytes_in(self):
        self.assertTrue(compare_submission_to_answer(set_index_col(self.dataset, self.dataset["bytes_in"]).index.to_list(), self.dataset["bytes_in"].to_list(),"set_index_col (bytes_in)"))
    
    def test_set_index_col_bytes_out(self):
        self.assertTrue(compare_submission_to_answer(set_index_col(self.dataset, self.dataset["bytes_out"]).index.to_list(), self.dataset["bytes_out"].to_list(),"set_index_col (bytes_out)"))
    
    def test_set_index_col_time(self):
        self.assertTrue(compare_submission_to_answer(set_index_col(self.dataset, self.dataset["time"]).index.to_list(), self.dataset["time"].to_list(),"set_index_col (time)"))
        
    def test_set_index_col_flagged(self):
        self.assertTrue(compare_submission_to_answer(set_index_col(self.dataset, self.dataset["flagged"]).index.to_list(), self.dataset["flagged"].to_list(),"set_index_col (flagged)"))
        
class Test_reset_index_col(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if folder_loc=="main":
            folder_path = os.path.join(os.getcwd(),"task1")
        else:
            folder_path = os.path.join(os.getcwd(),"..","task1")
        try:
            cls.dataset = pd.read_csv(os.path.join(folder_path,"reset_index_input.csv"), index_col=0)
        except:
            print("Test_reset_index_col Missing input files")
        try:
            cls.answer = pd.read_pickle(os.path.join(folder_path,"pkl_files","reset_index_col.pkl"))
        except:
            print("Test_reset_index_col Missing answer files")
    def test_reset_index_col(self):
        self.assertTrue(compare_submission_to_answer_df(reset_index_col(self.dataset),self.answer,"reset_index_col"))

class Test_set_col_type(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if folder_loc=="main":
            folder_path = os.path.join(os.getcwd(),"task1")
        else:
            folder_path = os.path.join(os.getcwd(),"..","task1")
        try:
            cls.dataset = pd.read_csv(os.path.join(folder_path,"sample.csv"),index_col=0)
        except:
            print("Test_set_col_type Missing input files")
        cls.target_col = "flagged"
    def test_set_col_type_target(self):
        self.assertEqual(np.dtype(set_col_type(self.dataset, self.target_col,np.float64)[self.target_col]), np.float64)
            
class Test_make_df_from_2d_array(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.array_2d = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14],[15,16,17],[18,19,20],[21,22,23],[24,25,26],[27,28,29]])
        cls.column_names = ["A","B","C"]
        cls.index = pd.Series([0,5,10,15,20,25,30,35,40,45])
            
        if folder_loc=="main":
            folder_path = os.path.join(os.getcwd(),"task1")
        else:
            folder_path = os.path.join(os.getcwd(),"..","task1")
        try:
            cls.make_df_from_2d_array_ans = pd.read_pickle(os.path.join(folder_path,"pkl_files","make_DF_from_2d_array.pkl"))
        except:
            print("Test_make_df_from_2d_array Missing answer files")
    def test_make_df_from_2d_array(self):
        self.assertTrue(compare_submission_to_answer_df(make_DF_from_2d_array(self.array_2d,self.column_names,self.index),self.make_df_from_2d_array_ans,"make_DF_from_2d_array"))

class Test_sort_df_by_column(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if folder_loc=="main":
            folder_path = os.path.join(os.getcwd(),"task1")
        else:
            folder_path = os.path.join(os.getcwd(),"..","task1")
        try:
            cls.dataset = pd.read_csv(os.path.join(folder_path,"sample.csv"),index_col=0)
            cls.dataset["time"] = pd.to_datetime(cls.dataset["time"])
        except:
            print("Test_sort_df_by_column Missing input files")
        try: 
            cls.sort_df_by_column_bytes_in_ans = pd.read_pickle(os.path.join(folder_path,"pkl_files","sort_bytes_in_desc.pkl"))
            cls.sort_df_by_column_src_ip_ans = pd.read_pickle(os.path.join(folder_path,"pkl_files","sort_src_ip_asc.pkl"))
        except:
            print("Test_sort_df_by_column Missing answer files")
        cls.sort_col1 = "bytes_in"
        cls.sort_col2 = "src_ip"
    def test_sort_df_bytes_in(self):
        self.assertTrue(compare_submission_to_answer_df(sort_DF_by_column(self.dataset, self.sort_col1, True),self.sort_df_by_column_bytes_in_ans,"sort_DF_by_column (bytes_in)"))
    
    def test_sort_df_src_ip(self):
        self.assertTrue(compare_submission_to_answer_df(sort_DF_by_column(self.dataset, self.sort_col2, False),self.sort_df_by_column_src_ip_ans,"sort_DF_by_column (src_ip)"))

class Test_drop_na_cols(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if folder_loc=="main":
            folder_path = os.path.join(os.getcwd(),"task1")
        else:
            folder_path = os.path.join(os.getcwd(),"..","task1")
        try:
            cls.dataset2 = pd.read_csv(os.path.join(folder_path,"sample2.csv"), index_col=0)
        except:
            print("Test_drop_na_cols Missing input files")
        try:
            cls.drop_na_cols_ans =pd.read_pickle(os.path.join(folder_path,"pkl_files","drop_NA_cols.pkl"))
        except:
            print("Test_drop_na_cols Missing answer files")
    def test_drop_na_cols(self):
        self.assertTrue(compare_submission_to_answer_df(drop_NA_cols(self.dataset2),self.drop_na_cols_ans,"drop_NA_cols"))

class Test_drop_na_rows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if folder_loc=="main":
            folder_path = os.path.join(os.getcwd(),"task1")
        else:
            folder_path = os.path.join(os.getcwd(),"..","task1")
        try:
            cls.dataset2 = pd.read_csv(os.path.join(folder_path,"sample2.csv"),index_col=0)
        except:
            print("Test_drop_na_rows Missing input files")
        try:
            cls.drop_na_rows_ans =pd.read_pickle(os.path.join(folder_path,"pkl_files","drop_NA_rows.pkl"))
        except:
            print("Test_drop_na_rows Missing answer files")
    def test_drop_na_rows(self):
        self.assertTrue(compare_submission_to_answer_df(drop_NA_rows(self.dataset2),self.drop_na_rows_ans,"drop_NA_rows"))
            
class Test_make_new_column(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if folder_loc=="main":
            folder_path = os.path.join(os.getcwd(),"task1")
        else:
            folder_path = os.path.join(os.getcwd(),"..","task1")
        try:
            cls.dataset = pd.read_csv(os.path.join(folder_path,"sample.csv"),index_col=0)
            cls.dataset["time"] = pd.to_datetime(cls.dataset["time"])
        except:
            print("Test_make_new_column Missing input files")
        try:
            cls.make_new_column_ans = pd.read_pickle(os.path.join(folder_path,"pkl_files","make_new_column.pkl"))
        except:
            print("Test_make_new_column Missing answer files")
        cls.new_column_name = "New Column"
        cls.new_column_values = [12,13,14,15,16,17,18,19,20,21]
    def test_make_new_column(self):
        self.assertTrue(compare_submission_to_answer_df(make_new_column(self.dataset, self.new_column_name, self.new_column_values),self.make_new_column_ans,"make_new_column"))

class Test_left_merge_DFs_by_column(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if folder_loc=="main":
            folder_path = os.path.join(os.getcwd(),"task1")
        else:
            folder_path = os.path.join(os.getcwd(),"..","task1")
        try:
            cls.dataset = pd.read_csv(os.path.join(folder_path,"sample.csv"),index_col=0)
            cls.dataset["time"] = pd.to_datetime(cls.dataset["time"])
            cls.dataset2 = pd.read_csv(os.path.join(folder_path,"sample2.csv"),index_col=0)
        except:
            print("Test_left_merge_DFs_by_column Missing input files")
        try:
            cls.left_merge_dfs_by_column_ans = pd.read_pickle(os.path.join(folder_path,"pkl_files","left_merge_DFs_by_column.pkl"))
        except:
            print("Test_left_merge_DFs_by_column Missing answer files")
        cls.merge_col = "src_ip"
    def test_left_merge_DFs_by_column(self):
        self.assertTrue(compare_submission_to_answer_df(left_merge_DFs_by_column(self.dataset, self.dataset2, self.merge_col),self.left_merge_dfs_by_column_ans,"left_merge_DFs_by_column"))

class Test_simpleClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.length = 1 
        cls.width = 2
        cls.height = 3
        cls.simpleClass = simpleClass(length=cls.length, width=cls.width, height=cls.height)
    def test_simpleClass_length(self):
        self.assertEqual(self.simpleClass.length, self.length)
        
    def test_simpleClass_width(self):
        self.assertEqual(self.simpleClass.width, self.width)
    
    def test_simpleClass_height(self):
        self.assertEqual(self.simpleClass.height, self.height)
class Test_find_dataset_statistics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if folder_loc=="main":
            folder_path = os.path.join(os.getcwd(),"task1")
        else:
            folder_path = os.path.join(os.getcwd(),"..","task1")
        try:
            cls.dataset = pd.read_csv(os.path.join(folder_path,"sample.csv"),index_col=0)
        except:
            print("Test_find_dataset_statistics Missing input files")
        cls.target_col = "flagged"
        cls.n_records_answer=10
        cls.n_columns_answer=6
        cls.n_negative_answer=5
        cls.n_positive_answer=5
        cls.perc_positive_answer=50
        
        #Run Function Code
        cls.n_records,cls.n_columns,cls.n_negative,cls.n_positive,cls.perc_positive = find_dataset_statistics(cls.dataset,cls.target_col)
    
    def test_nrecords(self):
        self.assertTrue(compare_submission_to_answer(self.n_records,self.n_records_answer,"n_records"))
    
    def test_n_columns(self):
        self.assertTrue(compare_submission_to_answer(self.n_columns,self.n_columns_answer,"n_columns"))
    
    def test_n_negative(self):
        self.assertTrue(compare_submission_to_answer(self.n_negative,self.n_negative_answer,"n_negative"))
    
    def test_n_positive(self):
        self.assertTrue(compare_submission_to_answer(self.n_positive,self.n_positive_answer,"n_positive"))
        
    def test_perc_positive(self):
        self.assertTrue(compare_submission_to_answer(self.perc_positive,self.perc_positive_answer,"perc_positive"))

if __name__ == '__main__':
    unittest.main()

