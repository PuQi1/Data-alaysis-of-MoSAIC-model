import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score



def calculate_age(file_path):
    """
    Load a CSV file, standardize its column names, and calculate ages of each clone ID in days.

    Parameters:
    file_path (str): The path to the CSV file to be processed.

    Returns:
    DataFrame: A DataFrame with standardized columns and calculated ages.
    """
    # Load the CSV file
    dataset = pd.read_csv(file_path)
    
    # Standardize column names
    dataset.columns = ['Time', 'Agebin', 'Clone ID', 'Frequency']
    
    # Convert Frequency to integer type if necessary
    if dataset['Frequency'].dtype != 'int64':
        dataset['Frequency'] = dataset['Frequency'].astype('int64')
    
    # Calculate ages of each clone ID in days
    dataset['age_in_days'] = round((dataset['Time'] * 10 - dataset['Agebin']) * 0.1, 1)
    
    return dataset


def cluster_for_top_n_age(file_path, time, num):
    """
    Perform clustering for the top N clone IDs based on frequency for different ages at a specific time.

    Parameters:
    file_path (str): The path to the CSV file to be processed.
    time (int): The specific time point to consider for clustering.
    num (int): The number of top clone IDs to consider based on frequency.

    Returns:
    The plot of silhouette scores for different numbers of clusters.
    """
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    dataset = calculate_age(file_path)
    # Calculate the sum of frequencies for each clone ID at different ages at a given time point
    freq_sum_age = dataset[dataset['Time'] == time].groupby(['age_in_days', 'Clone ID'])['Frequency'].sum().reset_index()
    # Find the top N clone IDs based on frequency
    age_top = freq_sum_age.groupby('Clone ID')['Frequency'].sum().nlargest(num).index
    # Filter the data for the top N clone IDs
    freq_sum_age = freq_sum_age[freq_sum_age['Clone ID'].isin(age_top)]
    # Pivot the DataFrame to have clone IDs as rows and ages as columns
    merged_df = freq_sum_age.pivot(index='Clone ID', columns='age_in_days', values='Frequency').fillna(0)

    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(merged_df)

    # Range of clusters to try
    range_n_clusters = list(range(2, 11))

    # Empty list to store silhouette scores
    silhouette_scores = []

    # Applying k-means and silhouette score calculation for different numbers of clusters
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(df_scaled)
        silhouette_avg = silhouette_score(df_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

    # Plot silhouette scores for different numbers of clusters
    plt.figure(figsize=(10, 6))
    plt.plot(range_n_clusters, silhouette_scores, marker='o')
    plt.title(f'Silhouette Scores (for Different Numbers of Clusters) for Top {num} Clone Ids in {file_name} at Day {time}')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    # Save the plot to the 'static' directory
    plot_path = 'static/cluster_age_plot1.png'  # Define the path to save the plot
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free up memory


def tsne_visualize_for_top_n_age(file_path, time, num, n_cluster):
    """
    Perform t-SNE visualization (the purpose is to reduce the data's dimension and plot on a 2-D figure) 
    for the top N clone IDs at a specific time with the given number of cluster.

    Parameters:
    file_path (str): The path to the CSV file to be processed.
    time (int): The specific time point to consider for clustering.
    num (int): The number of top clone IDs to consider based on frequency.
    n_cluster (int): The number of cluster to use for k-means clustering.

    Returns:
    The plot of the t-SNE visualization.
    """
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    dataset = calculate_age(file_path)
    # Calculate the sum of frequencies for each clone ID at different ages at a given time point
    freq_sum_age = dataset[dataset['Time'] == time].groupby(['age_in_days', 'Clone ID'])['Frequency'].sum().reset_index()
    age_top = freq_sum_age.groupby('Clone ID')['Frequency'].sum().nlargest(num).index
    # Filter the data for the top N clone IDs
    freq_sum_age = freq_sum_age[freq_sum_age['Clone ID'].isin(age_top)]
    merged_df = freq_sum_age.pivot(index='Clone ID', columns='age_in_days', values='Frequency').fillna(0)

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(merged_df)

    # Final k-means clustering with the best number of clusters
    final_clusterer = KMeans(n_clusters=n_cluster, random_state=10)
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
    plt.title(f'2D t-SNE Visualization of {n_cluster} Clusters for Top {num} Clone Ids in {file_name} at Day {time}')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    # Save the plot to the 'static' directory
    plot_path = 'static/cluster_age_plot2.png'  
    plt.savefig(plot_path)
    plt.close()  

#file_path = 'data/memslow.csv'
#num=1000
#time=90
#tsne_visualize_for_top_n_age(file_path, time, num, 2)