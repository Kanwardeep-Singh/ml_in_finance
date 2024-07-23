import unittest
from src.finance_ml.optimal_number_of_clusters import optimal_cluster_numbers
import pandas as pd
import numpy as np

class TestFinanceML(unittest.TestCase):

    def setUp(self):
        self.file_path = 'data/equities/AAPL_2020-04-07_2022-04-06.parquet'
        self.apple_data = optimal_cluster_numbers.optinal_cluster_numbers.load_data(self.file_path)

        # Take the first 2000 entries from the dataset
        self.apple_data = self.apple_data.head(2000)
        self.df_apple = pd.DataFrame({'clusters': [1, 2, 2, 1, 2, 1, 1, 2, 2, 1]})
        self.df_apple = optimal_cluster_numbers.optinal_cluster_numbers.preprocess_data(self.apple_data)

        

    def test_load_data(self):
        self.assertIsNotNone(self.apple_data)
        self.assertIsInstance(self.apple_data, pd.DataFrame)

    def test_preprocess_data(self):
        processed_data = optimal_cluster_numbers.optinal_cluster_numbers.preprocess_data(self.apple_data)
        self.assertIsNotNone(processed_data)
        # Add more specific assertions based on your preprocessing logic

    def test_kmeans_silhouette_scores(self):
        k_values = range(2, 20)
        silhouette_scores = optimal_cluster_numbers.optinal_cluster_numbers.kmeans_silhouette_scores(self.df_apple, k_values)
        self.assertEqual(len(silhouette_scores), len(k_values))

    def test_kmeans_davies_bouldin_score(self):
        k_values = range(2, 20)
        davies_bouldin_score = optimal_cluster_numbers.optinal_cluster_numbers.kmeans_davies_bouldin_score(self.df_apple, k_values)
        self.assertEqual(len(davies_bouldin_score), len(k_values))



    def test_silhouette_score_visualization(self):
        k_values = range(5,8)
        with self.assertRaises(Exception):
            optimal_cluster_numbers.optinal_cluster_numbers.silhouette_score_visualization(self.df_apple, k_values)

    
if __name__ == '__main__':
    unittest.main()