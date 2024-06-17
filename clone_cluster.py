import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


def load_and_count_frequency(file_path):
    """
    Load a CSV file, standardize its column names, and summarize the 'Frequency' data.

    Parameters:
    file_path (str): The path to the CSV file to be processed.

    Returns:
    DataFrame: A DataFrame with the sum of 'Frequency' grouped by 'Time' and 'Clone ID'.
    """
    # Load the CSV file
    dataset = pd.read_csv(file_path)
    
    # Standardize column names
    dataset.columns = ['Time', 'Agebin', 'Clone ID', 'Frequency']
    
    # Convert Frequency to integer type if necessary
    if dataset['Frequency'].dtype != 'int64':
        dataset['Frequency'] = dataset['Frequency'].astype('int64')
    
    # Group by 'Time' and 'Clone ID' and summarize 'Frequency'
    frequency_summary = dataset.groupby(['Time', 'Clone ID'])['Frequency'].sum().reset_index()
    
    return frequency_summary

def cluster_for_top_n(file_path, num):
    """
    Perform clustering for the top N clone IDs based on frequency for different time points..

    Parameters:
    file_path (str): The path to the CSV file to be processed.
    num (int): The number of top clone IDs to consider based on frequency.

    Returns:
    The plot of silhouette scores for different numbers of clusters.
    """
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    # Load the data and summarize the 'Frequency' data
    freq_sum = load_and_count_frequency(file_path)
    day_top = freq_sum.groupby('Clone ID')['Frequency'].sum().nlargest(num).index
    # Filter the data for the top N clone IDs
    freq_sum_day = freq_sum[freq_sum['Clone ID'].isin(day_top)]
    # Pivot the DataFrame to have clone IDs as rows and time points as columns
    merged_df = freq_sum_day.pivot(index='Clone ID', columns='Time', values='Frequency').fillna(0)

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(merged_df)

    # Range of clusters to try
    range_n_clusters = list(range(2, 11))

    # Empty list to store silhouette scores
    silhouette_scores = []

    # Applying k-means and silhouette score calculation
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(df_scaled)
        silhouette_avg = silhouette_score(df_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(range_n_clusters, silhouette_scores, marker='o')
    plt.title(f'Silhouette Scores (for Different Numbers of Clusters) for Top {num} Clone Ids in {file_name}')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    # Save the plot to the 'static' directory
    plot_path = 'static/cluster_plot1.png'  
    plt.savefig(plot_path)
    plt.close()  


def tsne_visualize_for_top_n(file_path, num, n_clusters):
    """
    Perform t-SNE visualization (the purpose is to reduce the data's dimension and plot on a 2-D figure)
    for the top N clone IDs with the given number of clusters.

    Parameters:
    file_path (str): The path to the CSV file to be processed.
    num (int): The number of top clone IDs to consider based on frequency.
    n_clusters (int): The number of clusters to use for k-means clustering.

    Returns:
    The plot of the t-SNE visualization.
    """
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    freq_sum = load_and_count_frequency(file_path)
    day_top = freq_sum.groupby('Clone ID')['Frequency'].sum().nlargest(num).index
    # Filter the data for the top N clone IDs
    freq_sum_day = freq_sum[freq_sum['Clone ID'].isin(day_top)]
    # Pivot the DataFrame to have clone IDs as rows and time points as columns
    merged_df = freq_sum_day.pivot(index='Clone ID', columns='Time', values='Frequency').fillna(0)

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(merged_df)

    # Final k-means clustering with the best number of clusters
    final_clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    final_cluster_labels = final_clusterer.fit_predict(df_scaled)

    # Optionally add cluster labels back to the original DataFrame
    merged_df['Cluster'] = final_cluster_labels

    # Splitting features and labels
    X = merged_df.drop('Cluster', axis=1)
    y = merged_df['Cluster']

    # Standardizing the features (important for t-SNE)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6)

    plt.colorbar(scatter, label='Cluster')

    plt.title(f'2D t-SNE Visualization of {n_clusters} Clusters for Top {num} Clone Ids in {file_name}')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    # Save the plot to the 'static' directory
    plot_path = 'static/cluster_plot2.png'  
    plt.savefig(plot_path)
    plt.close()  
    





#file_path = 'data/memslow.csv'
#num=5000
#cluster_for_top_n(file_path, 1000)