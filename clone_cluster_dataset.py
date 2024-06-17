import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
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


def create_overlap_df(file_paths, time_value):
    """
    Create a DataFrame showing the frequencies of each Clone Id for given CSV files at a specific time.

    Parameters:
    file_paths (list of str): A list of paths to the CSV files.
    time_value (int): The specific time value to filter the data.

    Returns:
    DataFrame: A DataFrame with the frequencies of Clone IDs at each file.
    Rows are Clone IDs and columns are the file labels.
    """
    data_frames = []
    
    for file_path in file_paths:
        freq_sum_day = load_and_count_frequency(file_path)
        
        # Filter data for the given time value
        filtered_df = freq_sum_day[freq_sum_day['Time'] == time_value]
        
        # Drop the 'Time' column
        filtered_df.drop('Time', axis=1, inplace=True)
        
        # Rename the 'Frequency' column for each DataFrame
        file_label = os.path.splitext(os.path.basename(file_path))[0]
        filtered_df.rename(columns={"Frequency": f"{file_label}"}, inplace=True)
        # Append each file's dataFrame to the list
        data_frames.append(filtered_df)
    
    # Merge all DataFrames on 'Clone ID'
    df_final = reduce(lambda left, right: pd.merge(left, right, on='Clone ID', how='outer'), data_frames)
    
    # Fill NaN values with 0
    df_final.fillna(0, inplace=True)
    
    # Set 'Clone ID' as the index
    df_final.set_index('Clone ID', inplace=True)
    
    return df_final


def plot_scatter_nonzero_ids(file_paths, time_value):
    """
    Plot scatter plot of non-zero Clone IDs by log(Clone ID) for a given set of CSV files and time value.

    Parameters:
    file_paths (list of str): A list of paths to the CSV files.
    time_value (int): The specific time value to filter the data.
    
    Returns:
    Plot whether each Clone ID appears in each dataset (by log(Clone ID)) for a given set of CSV files and time value.
    """
    # Create the frequency DataFrame
    df_final_time = create_overlap_df(file_paths, time_value)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    # Select the non-zero Clone IDs for each dataset
    for i, column in enumerate(df_final_time.columns):
        # Get the non-zero Clone IDs
        non_zero_mask = df_final_time[column] != 0
        clone_ids = df_final_time.index[non_zero_mask]
        log_clone_ids = np.log(clone_ids)

        y_values = np.full(log_clone_ids.shape, i + 1)

        ax.scatter(log_clone_ids, y_values, alpha=0.5, label=column)

    ax.legend(title='Dataset Type', loc='lower left')
    ax.set_xlabel('Log(Clone ID)')
    ax.set_ylabel('Dataset Category')
    ax.set_title(f'Scatter Plot of Non-Zero Frequencies by Log(Clone ID) on Day {time_value} for Selected Datasets')
    ax.set_yticks(range(1, len(df_final_time.columns) + 1))
    ax.set_yticklabels(df_final_time.columns)
    ax.grid(True)  # Add grid

    # Save the plot to the 'static' directory
    plot_path = 'static/cluster_dataset_plot1.png'  
    plt.savefig(plot_path)
    plt.close()  


def plot_scatter_frequencies(file_paths, time_value):
    """
    Plot scatter plot of frequencies of each Clone ID (by log(Clone ID)) for a given set of CSV files and time value.

    Parameters:
    file_paths (list of str): A list of paths to the CSV files.
    time_value (int): The specific time value to filter the data.
    
    Returns:
    Plot the frequencies of each Clone ID (by log(Clone ID)).
    """
    # Create the frequency DataFrame
    df_final_time = create_overlap_df(file_paths, time_value)
    x_values = np.log(df_final_time.index)

    fig, ax = plt.subplots(figsize=(10, 6))
    # Select the non-zero Clone IDs for each dataset and abstract the frequencies
    for column in df_final_time.columns:
        non_zero_mask = df_final_time[column] != 0
        non_zero_data = df_final_time.loc[non_zero_mask, column]

        ax.scatter(x_values[non_zero_mask], non_zero_data, alpha=0.6, label=column)

    ax.legend(title='Dataset Type', loc='upper right')
    ax.set_xlabel('Log(Clone ID)')
    ax.set_ylabel('Frequency Value')
    ax.set_title(f'Scatter Plot of Frequencies by Log(Clone ID) on Day {time_value} for Selected Datasets')
    ax.grid(True)  # Add grid

    # Save the plot to the 'static' directory
    plot_path = 'static/cluster_dataset_plot2.png' 
    plt.savefig(plot_path)
    plt.close()  


def analyze_clusters(file_paths, time_value):
    """
    Perform clustering for the clone IDs based on frequency in different datasets at a specific time.

    Parameters:
    file_paths (list of str): A list of paths to the CSV files.
    time_value (int): The specific time value to filter the data.
    
    Returns:
    The plot of the silhouette scores for different number of clusters.
    """
    # Create the frequency DataFrame
    df_final_time = create_overlap_df(file_paths, time_value)
    
    # Standardize the data
    scaler = StandardScaler()
    df_scaled_time = scaler.fit_transform(df_final_time)

    # Range of clusters to try
    range_n_clusters = list(range(2, 11))

    # Empty list to store silhouette scores
    silhouette_scores = []

    # Applying k-means and silhouette score calculation
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(df_scaled_time)
        silhouette_avg = silhouette_score(df_scaled_time, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(range_n_clusters, silhouette_scores, marker='o')
    plt.title(f'Silhouette Scores (for Different Numbers of Clusters) for Selected Datasets on Day {time_value}')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)  # Add grid
    # Save the plot to the 'static' directory
    plot_path = 'static/cluster_dataset_plot3.png'  
    plt.savefig(plot_path)
    plt.close()  


def tsne_visualization(file_paths, time_value, n_clusters):
    """
    Perform t-SNE visualization (the purpose is to reduce the data's dimension and plot on a 2-D figure)
    for the given number of clusters.

    Parameters:
    file_paths (list of str): A list of paths to the CSV files.
    time_value (int): The specific time value to filter the data.
    n_clusters (int): The number of clusters to use for k-means clustering.

    Returns:
    The plot of the t-SNE visualization.
    """
    df_final = create_overlap_df(file_paths, time_value)
    
    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_final)

    # Final k-means clustering with the best number of clusters
    final_clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    final_cluster_labels = final_clusterer.fit_predict(df_scaled)

    # Optionally add cluster labels back to the original DataFrame
    df_final['Cluster'] = final_cluster_labels

    # Splitting features and labels
    X = df_final.drop('Cluster', axis=1)
    y = df_final['Cluster']

    # Standardizing the features (important for t-SNE)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6)

    plt.colorbar(scatter, label='Cluster')

    plt.title(f'2D t-SNE Visualization of {n_clusters} Clusters for Selected Datasets on Day {time_value}')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    # Save the plot to the 'static' directory
    plot_path = 'static/cluster_dataset_plot4.png'  
    plt.savefig(plot_path)
    plt.close()  
    

#file_paths = ['data/memfast.csv', 'data/memslow.csv'] 
#num=1000
#time=270
#plot_scatter_nonzero_ids(file_paths, time)

