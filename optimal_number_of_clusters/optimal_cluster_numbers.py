import sklearn
import numpy as np
import sys
import importlib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from yellowbrick.cluster import SilhouetteVisualizer

from sklearn.cluster import KMeans

from scipy.cluster import hierarchy


class optinal_cluster_numbers:
    def load_data(file_path):
        apple_data = pd.read_parquet(file_path)
        return apple_data
    

    def preprocess_data(data):
        features = ['VOLUME', 'VW', 'OPEN', 'CLOSE', 'HIGHT', 'LOW', 'TRANSACTIONS', 'DATE']

        # Ensure 'DATE' column is in Pandas datetime format
        data['DATE'] = pd.to_datetime(data['DATE'])

        # Convert 'DATE' column to NumPy array before subtraction
        data['DATE'] = (data['DATE'].values.astype(np.int64) - pd.Timestamp("1970-01-01").value) // int(
            1e9)  # Use int(1e9) instead of '1s'

        df_apple = data[features]
        return df_apple


    def kmeans_silhouette_scores(df, k_values):
        silhouette_scores = []

      
        for n_clusters in k_values:
            #iterating through cluster sizes
            clusterer = KMeans(n_clusters = n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(df)
                
            #Finding the average silhouette score
            silhouette_avg = silhouette_score(df, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            

        return silhouette_scores


    def kmeans_davies_bouldin_score(df, k_values):
        davies_bouldin_score = []

        for n_clusters in k_values:
            #iterating through cluster sizes
            clusterer = KMeans(n_clusters = n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(df)
        
        #Finding the Davies Bouldin Index score
            davies_bouldin= sklearn.metrics.davies_bouldin_score(df, cluster_labels)
            davies_bouldin_score.append(davies_bouldin) 

        return davies_bouldin_score



    class SomeSpecificException:
        pass


    def elbow_method_visualization(df, some_condition=None):
        try:
            wcss = [] 
            for i in range(2, 15): 
                kmeans = KMeans(n_clusters = i, random_state = 42)
                kmeans.fit(df) 
                wcss.append(kmeans.inertia_)

            plt.plot(range(2, 15), wcss)
            plt.title('Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS') 
            plt.show()

        except ValueError as e:
        # Catch the specific exception and re-raise it
            raise e


    def silhouette_score_visualization(df, k_values):
        ax = plt.subplots(2, 2, figsize=(15,8))
        for k in k_values:
            
            #Create KMeans instance for different number of clusters
            km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100, random_state=42)
            q, mod = divmod(k, 2)

            clusterer = KMeans(n_clusters=k, random_state=42)
            cluster_labels = clusterer.fit_predict(df)
            silhouette_avg = silhouette_score(df, cluster_labels)
            
            
            #Create SilhouetteVisualizer instance with KMeans instance Fit the visualizer
            visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-3][mod])
            visualizer.fit(df)
        
    def hierarchical_dendrogram(df):
        Z = hierarchy.linkage(df, 'single')
        plt.figure(figsize=(25, 15))
        dn = hierarchy.dendrogram(Z,leaf_font_size=8.)


    if __name__ == "__main__":
        # Example usage of the functions
        file_path = 'dataset.parquet'
        apple_data = load_data(file_path)
        df_apple = preprocess_data(apple_data)

        k_values = range(2, 31)
        silhouette_scores = kmeans_silhouette_scores(df_apple, k_values)
        calinski_scores = kmeans_davies_bouldin_score(df_apple, k_values)

        elbow_method_visualization(df_apple)
        silhouette_score_visualization(df_apple, k_values)
        hierarchical_dendrogram(df_apple)
