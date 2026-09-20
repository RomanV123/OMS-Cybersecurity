import numpy as np
import pandas as pd

def find_data_type(dataset:pd.DataFrame,column_name:str) -> np.dtype:
    # TODO: Read the function description in
    # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task1.html
    # and implement the function as described.

    column = dataset[column_name]
    return column.dtype

def set_index_col(dataset:pd.DataFrame,index:pd.Series) -> pd.DataFrame:
    # TODO: Read the function description in
    # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task1.html
    # and implement the function as described.
    dataset.index = index

    return dataset

def reset_index_col(dataset:pd.DataFrame) -> pd.DataFrame:
    # TODO: Read the function description in
    # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task1.html
    # and implement the function as described.

    result = dataset.reset_index(drop =True)

    return result

def set_col_type(dataset:pd.DataFrame,column_name:str,new_col_type:type) -> pd.DataFrame:
    # TODO: Read the function description in
    # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task1.html
    # and implement the function as described.

    dataset[column_name]=dataset[column_name].astype(new_col_type, copy=True)
    return dataset

def make_DF_from_2d_array(array_2d:np.array,column_name_list:list[str],index:pd.Series) -> pd.DataFrame:
    # TODO: Read the function description in
    # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task1.html
    # and implement the function as described.

    dataframe = pd.DataFrame(
        data = array_2d,
        columns = column_name_list,
        index = index
    )
    return dataframe

def sort_DF_by_column(dataset:pd.DataFrame,column_name:str,descending:bool) -> pd.DataFrame:
    # TODO: Read the function description in
    # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task1.html
    # and implement the function as described.

    dataset = dataset.sort_values(by = column_name, ascending = not descending)
    return dataset

def drop_NA_cols(dataset:pd.DataFrame) -> pd.DataFrame:
    # TODO: Read the function description in
    # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task1.html
    # and implement the function as described.

    dataset_NA = dataset.dropna(axis = 1 )
    return dataset_NA
    

def drop_NA_rows(dataset:pd.DataFrame) -> pd.DataFrame:
    # TODO: Read the function description in
    # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task1.html
    # and implement the function as described.

    dataset_NA = dataset.dropna(axis = 0 )
    return dataset_NA

def make_new_column(dataset:pd.DataFrame,new_column_name:str,new_column_value:list) -> pd.DataFrame:
    # TODO: Read the function description in
    # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task1.html
    # and implement the function as described.

    dataset[new_column_name]= new_column_value
    return dataset

def left_merge_DFs_by_column(left_dataset:pd.DataFrame,right_dataset:pd.DataFrame,join_col_name:str) -> pd.DataFrame:
    # TODO: Read the function description in
    # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task1.html
    # and implement the function as described.
    left_merged_df = left_dataset.merge(
        right_dataset,
        how = 'left',
        on = join_col_name
    )
    

    return left_merged_df

class simpleClass():
    # TODO: Read the function description in
    # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task1.html
    # and implement the function as described.
    def __init__(self, length:int, width:int, height:int):

        self.length = length
        self.width = width
        self.height = height


def find_dataset_statistics(dataset:pd.DataFrame,label_col:str) -> tuple[int,int,int,int,int]:
    # TODO: Read the function description in
    # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task1.html
    # and implement the function as described.

    n_records = int(dataset.shape[0])
    n_columns = int(dataset.shape[1])
    n_negative = int((dataset[label_col] == 0 ).sum())
    n_positive = int((dataset[label_col] == 1 ).sum())
    perc_positive =  int((n_positive / n_records)*100)

    return n_records,n_columns,n_negative,n_positive,perc_positive
