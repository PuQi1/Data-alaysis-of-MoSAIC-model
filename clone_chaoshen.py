import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr



def load_and_count_frequency(file_path):
    """
    Load a CSV file, standardize its column names, and summarize the 'Frequency' data.

    Parameters:
    file_path (str): The path to the CSV file to be processed.

    Returns:
    tuple: A tuple containing:
        - frequency_summary (DataFrame): A DataFrame with the sum of 'Frequency' grouped by 'Time' and 'Clone ID'.
        - frequency_summary_all (DataFrame): A DataFrame with the sum of 'Frequency' grouped by 'Time'.
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
    frequency_summary_all = dataset.groupby(['Time'])['Frequency'].sum().reset_index()
    
    return frequency_summary, frequency_summary_all


# function of chaoshen entropy
def entropy_chao_shen(y, unit='log'):
    """
    Calculate the Chao-Shen entropy for a given set of counts.

    Parameters:
    y (array-like): Array of counts.
    unit (str): The logarithmic unit to use ('log', 'log2', 'log10').

    Returns:
    float: The calculated Chao-Shen entropy.
    """
    # Validate the unit parameter
    if unit not in ['log', 'log2', 'log10']:
        raise ValueError("Invalid unit. Choose from 'log', 'log2', or 'log10'.")
    yx = y[y > 0]           # remove bins with zero counts
    n = np.sum(yx)          # total number of counts
    p = yx / n              # empirical frequencies
    f1 = np.sum(yx == 1)    # number of singletons
    if f1 == n:
        f1 = n - 1           # avoid C=0
    C = 1 - f1 / n          # estimated coverage
    pa = C * p              # coverage adjusted empirical frequencies
    la = (1 - (1 - pa)**n)  # probability to see a bin (species) in the sample
    H = -np.sum(pa * np.log(pa) / la)  # Chao-Shen (2003) entropy estimator
    if unit == 'log2':
        H /= np.log(2)      # change from log to log2 scale
    elif unit == 'log10':
        H /= np.log(10)     # change from log to log10 scale
    return H


# calculate chaoshen entropy for each time point
def calculate_chaoshen_index(freq_sum_day):
    """
    Calculate the Chao-Shen entropy for each time point.

    Parameters:
    freq_sum_day (DataFrame): DataFrame with 'Time' and 'Frequency' columns.

    Returns:
    DataFrame: A DataFrame with 'Time' and 'ChaoShen' columns.
    """
    
    time_points = freq_sum_day['Time'].unique()
    # Calculate Chao-Shen entropy for each time point
    chao_shen_day = []
    for t in time_points:
        time_data = freq_sum_day[freq_sum_day['Time'] == t]
        if not time_data.empty:
            # Calculate Chao-Shen entropy
            chao_shen = entropy_chao_shen(time_data['Frequency'])
            chao_shen_day.append({'Time': t, 'ChaoShen': chao_shen})
    return pd.DataFrame(chao_shen_day)


def plot_chaoshen_indices(file_paths):
    """
    Plot the Chao-Shen indices for multiple datasets over time.

    Parameters:
    file_paths (list of str): List of file paths to the CSV files to be processed.
    
    Returns:
    The plot of Chao-Shen indices over times.
    """
    plt.figure(figsize=(10, 6))
    
    for file_path in file_paths:
        # Load the data and calculate the Chao-Shen index
        frequency_data = load_and_count_frequency(file_path)[0]
        chaoshen_indices = calculate_chaoshen_index(frequency_data)
        
        times = list(chaoshen_indices['Time'])
        indices = list(chaoshen_indices['ChaoShen'])
        
        label = os.path.splitext(os.path.basename(file_path))[0]
        plt.plot(times, indices, marker='o', label=label)
    
    plt.xlabel('Time')
    plt.ylabel('Chaoshen Index')
    plt.title('Chaoshen Index Over Time for Selected Datasets')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to the 'static' directory
    plot_path = 'static/chaoshen_indices_plot.png'  
    plt.savefig(plot_path)
    plt.close() 



def calculate_pearson_chaoshen(file_paths):
    """
    Calculate the Pearson correlation between Chao-Shen entropy and total frequency.

    Parameters:
    file_paths (list of str): List of file paths to the CSV files to be processed.

    Returns:
    dict: A dictionary with file names as keys and Pearson correlation tuples as values.
    """
    correlation = {}
    
    for file_path in file_paths:
        # Load the data and calculate the Chao-Shen index
        frequency_data = load_and_count_frequency(file_path)[0]
        chaoshen_indices = calculate_chaoshen_index(frequency_data)
        # Calculate the total frequency for each given time point
        frequency_all_data = load_and_count_frequency(file_path)[1]
        # Calculate the Pearson correlation between Chao-Shen entropy and total frequency
        Pearson_correlation = pearsonr(chaoshen_indices['ChaoShen'], frequency_all_data['Frequency'])
        
        name = os.path.splitext(os.path.basename(file_path))[0]
        correlation[name] = Pearson_correlation
    
    return correlation
    

