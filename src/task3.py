import numpy as np
import pandas as pd
from typing import Optional
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

class KmeansClustering:
    def __init__(self,random_state: int, init: str, n_init: int, max_iter: int, algorithm: str,tol: float):
        # TODO: Read the function description in
        # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task3.html
        # and implement the function as described.

        self.random_state = random_state
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.algorithm = algorithm
        self.tol = tol


        pass

    def kmeans_get_n_clusters(self, train_features:pd.DataFrame) -> int:
        # TODO: Read the function description in
        # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task3.html
        # and implement the function as described.
        self.kmeans = KMeans(
            init = self.init,
            n_init = self.n_init,
            max_iter= self.max_iter,
            random_state = self.random_state,
            algorithm = self.algorithm,
            tol = self.tol
        )
        visualizer = KElbowVisualizer(self.kmeans, k=(2,10))
        visualizer.fit(train_features)
        best_k = int(visualizer.elbow_value_)
        
        return best_k

    def kmeans_train(self, train_features: pd.DataFrame, k:Optional[int]=None) -> list:  # noqa: UP045
        # TODO: Read the function description in
        # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task3.html
        # and implement the function as described.
        
        
        if k is None:
            k = self.kmeans_get_n_clusters(train_features)

        self.kmeans = KMeans(
            n_clusters = k,
            init = self.init,
            n_init = self.n_init,
            max_iter = self.max_iter,
            random_state = self.random_state,
            algorithm = self.algorithm,
            tol = self.tol
        )

        cluster_ids = self.kmeans.fit_predict(train_features)
        return cluster_ids.tolist()

    def kmeans_test(self, test_features: pd.DataFrame) -> list:
        # TODO: Read the function description in
        # https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task3.html
        # and implement the function as described.


        cluster_ids = self.kmeans.predict(test_features)

        return cluster_ids.tolist()
