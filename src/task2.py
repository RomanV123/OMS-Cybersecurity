import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def tts(  dataset: pd.DataFrame,
                       label_col: str, 
                       test_size: float,
                       should_stratify: bool,
                       random_state: int) -> tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
    # TODO: Read the function description in
    # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html
    # and implement the function as described.
    features = dataset.drop(columns=[label_col])
    labels = dataset[label_col]

    if should_stratify:
        stratify_values = labels
    else:
        stratify_values = None
    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size = test_size,
        random_state = random_state,
        stratify = stratify_values
    )
    return train_features,test_features,train_labels,test_labels

class PreprocessDataset:
    def __init__(self, 
                 one_hot_encode_cols:list[str],
                 min_max_scale_cols:list[str],
                 n_components:int,
                 feature_engineering_functions:dict
                 ):
        # TODO: Add any state variables you may need to make your functions work

        self.one_hot_encode_cols = one_hot_encode_cols
        self.min_max_scale_cols = min_max_scale_cols
        self.n_components = n_components
        self.feature_engineering_functions = feature_engineering_functions

        self.one_hot_encoder = OneHotEncoder(handle_unknown="ignore")
        self.min_max_scaler = MinMaxScaler()
        self.pca = PCA(n_components=self.n_components, random_state=0)
        self.pca_columns = None


        return

    def one_hot_encode_columns_train(self,train_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in
        # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html
        # and implement the function as described.

        columns_to_encode = train_features[self.one_hot_encode_cols]
        remaining_cols = train_features.drop(columns = self.one_hot_encode_cols)

        self.one_hot_encoder.fit(columns_to_encode)
        encoded_values = self.one_hot_encoder.transform(columns_to_encode)

        encoded_df = pd.DataFrame(
        encoded_values.toarray(),
        columns=self.one_hot_encoder.get_feature_names_out(),
        index=train_features.index
    )
        one_hot_encoded_dataset = pd.concat([remaining_cols, encoded_df], axis = 1)

        return one_hot_encoded_dataset

    def one_hot_encode_columns_test(self, test_features: pd.DataFrame) -> pd.DataFrame:
        columns_to_encode = test_features[self.one_hot_encode_cols]
        remaining_cols = test_features.drop(columns=self.one_hot_encode_cols)

        encoded_values = self.one_hot_encoder.transform(columns_to_encode)

    # Support either sparse or dense encoder output.
        if hasattr(encoded_values, "toarray"):
            encoded_values = encoded_values.toarray()

        encoded_df = pd.DataFrame(
            encoded_values,
            columns=self.one_hot_encoder.get_feature_names_out(),
            index=test_features.index
    )

        return pd.concat([remaining_cols, encoded_df], axis=1)

    def min_max_scaled_columns_train(self,train_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in
        # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html
        # and implement the function as described.
        min_max_scaled_dataset = train_features.copy()

        if self.min_max_scale_cols:
            scaled_data = self.min_max_scaler.fit_transform(
                train_features[self.min_max_scale_cols]
            )
            min_max_scaled_dataset[self.min_max_scale_cols] = scaled_data
        return min_max_scaled_dataset

    def min_max_scaled_columns_test(self,test_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in
        # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html
        # and implement the function as described.

        min_max_scaled_dataset = test_features.copy()

        if self.min_max_scale_cols:
            scaled_data = self.min_max_scaler.transform(
                test_features[self.min_max_scale_cols]

            )

            min_max_scaled_dataset[self.min_max_scale_cols] = scaled_data


        return min_max_scaled_dataset

    def pca_train(self,train_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in
        # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html
        # and implement the function as described.
        clean_features = train_features.dropna(axis=1)

        self.pca_columns = clean_features.columns.tolist()

        pca_values = self.pca.fit_transform(clean_features)



        return pd.DataFrame(
        pca_values,
        columns = [
            f"component_{i}"
            for i in range(1,self.n_components+1)
        ],
        index = train_features.index
    )


    def pca_test(self,test_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in
        # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html
        # and implement the function as described.
        clean_features = test_features[self.pca_columns]


        pca_values = self.pca.transform(clean_features)
        return pd.DataFrame(
            pca_values,
            columns =[
                f"component_{i}"
                for i in range(1,self.n_components + 1)
            ],
            index=test_features.index
        )


    def feature_engineering_train(self,train_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in
        # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html
        # and implement the function as described.
        feature_engineered_dataset = train_features.copy()

        for column_name, calculate in self.feature_engineering_functions.items():
            feature_engineered_dataset[column_name] = calculate(train_features)

        return feature_engineered_dataset
    
    def feature_engineering_test(self,test_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in
        # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html
        # and implement the function as described.

        feature_engineered_dataset = test_features.copy()

        for column_name, feature_function in self.feature_engineering_functions.items():
            feature_engineered_dataset[column_name] = feature_function(test_features)

        return feature_engineered_dataset

    def preprocess_train(self,train_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in
        # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html
        # and implement the function as described.

        encoded_data = self.one_hot_encode_columns_train(train_features)
        scaled_columns = self.min_max_scaled_columns_train(encoded_data)
        feature_engineered_data = self.feature_engineering_train(scaled_columns)



        return feature_engineered_data
    
    def preprocess_test(self,test_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in
        # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html
        # and implement the function as described.
        encoded_data = self.one_hot_encode_columns_test(test_features)
        scaled_columns = self.min_max_scaled_columns_test(encoded_data)
        feature_engineered_data = self.feature_engineering_test(scaled_columns)


        
        return feature_engineered_data

