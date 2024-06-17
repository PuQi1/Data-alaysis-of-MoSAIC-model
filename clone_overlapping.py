import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from sklearn.preprocessing import scale
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from functools import reduce
from scipy.spatial.distance import pdist, squareform
import os

def process_dataset(file_path):
    """
    Process the dataset by loading the CSV file, standardizing column names, and calculating unique clone IDs per time point.

    Parameters:
    file_path (str): The path to the CSV file to be processed.

    Returns:
    tuple: A tuple containing:
        - unique_ids (DataFrame): DataFrame with unique Clone IDs and Time.
        - unique_ids_count (DataFrame): DataFrame with the count of unique Clone IDs per Time.
    """
    # Load the CSV file
    dataset = pd.read_csv(file_path)
    
    # Standardize column names
    dataset.columns = ['Time', 'Agebin', 'Clone ID', 'Frequency']
    
    # Drop duplicates and calculate unique clone IDs per time point
    unique_ids = dataset[['Time', 'Clone ID']].drop_duplicates()
    unique_ids_count = unique_ids.groupby('Time')['Clone ID'].nunique().reset_index()
    
    return unique_ids, unique_ids_count



def plot_common_clones_over_time(file_paths, output_dir='static'):
    """
    Create plots of common clone IDs over time.
    Plot 1 is how many common clone IDs are shared across selected datasets over time.
    Plot 2 is a scatter plot of shared Clone ID (by log(Clone ID)) vs Time.
    Plot 3 is the percentage of shared Clone IDs in each selected dataset across time.

    Parameters:
    file_paths (list of str): A list of paths to the CSV files.
    output_dir (str): The directory to save the plots.
    
    Returns:
    The three plots of common clone IDs over time.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare color map and pick colors based on the number of file_paths
    color_map = colormaps['tab10']
    colors = [color_map(i / len(file_paths)) for i in range(len(file_paths))]
    
    # Calculate unique IDs and their counts for each dataset
    dfs = []
    labels = []
    for file_path in file_paths:
        unique_ids, unique_ids_count = process_dataset(file_path)
        dfs.append((unique_ids, unique_ids_count))
        labels.append(os.path.splitext(os.path.basename(file_path))[0])
    
    # Merge all unique ID dataframes on ['Time', 'Clone ID']
    merged_df = dfs[0][0]
    for df, _ in dfs[1:]:
        merged_df = merged_df.merge(df, on=['Time', 'Clone ID'], how='inner')
    
    # Calculate the count of common Clone IDs over time
    time_clone_counts = merged_df.groupby('Time').size().reset_index(name='Clone ID Count')

    # Plot 1: Line plot of number of common Clone IDs over time
    plt.figure(figsize=(10, 6))
    plt.plot(time_clone_counts['Time'], time_clone_counts['Clone ID Count'], marker='o', color='blue')
    plt.title('Number of Shared Clone IDs Across Selected Datasets Over Time')
    plt.xlabel('Time')
    plt.ylabel('Number of Shared Clone IDs')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'over_plot1.png'))
    plt.close()
    
    # Plot 2: Scatter plot of shared Clone ID (by log(Clone ID)) vs Time
    if not merged_df.empty:
        # Adding logarithmic transformation to handle zero counts gracefully
        merged_df['log(Clone ID)'] = np.log(merged_df['Clone ID'])
        plt.figure(figsize=(10, 6))
        plt.scatter(merged_df['log(Clone ID)'], merged_df['Time'], color='blue')
        plt.ylabel('Time')
        plt.xlabel('log(Clone ID)')
        plt.title('Scatter Plot of Shared Log(Clone ID) vs Time')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'over_plot2.png'))
        plt.close()

    # Plot 3: Line plot of percentage of shared Clone IDs in each dataset over time
    plt.figure(figsize=(10, 6))
    for (_, unique_ids_count), label, color in zip(dfs, labels, colors):
        # Merge the time_clone_counts with unique_ids_count
        merged = pd.merge(time_clone_counts[['Time', 'Clone ID Count']], unique_ids_count, on='Time')
        # Calculate the percentage of shared Clone IDs
        merged[f'Percentage_{label}'] = (merged['Clone ID Count'] / merged['Clone ID']) * 100
        plt.plot(merged['Time'], merged[f'Percentage_{label}'], label=label, color=color)

    plt.xlabel('Time')
    plt.ylabel('Percentage of Shared Clone IDs')
    plt.title('Percentage of Shared Clone IDs Across Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'over_plot3.png'))
    plt.close()