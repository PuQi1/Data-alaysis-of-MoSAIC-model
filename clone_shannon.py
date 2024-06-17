import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
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


def calculate_shannon_index(frequency_summary):
    """
    Calculate the Shannon index for each time point.

    Parameters:
    frequency_summary (DataFrame): The DataFrame containing the frequency data.

    Returns:
    dict: A dictionary with time points as keys and Shannon index values as values.
    """
    shannon_indices = {}
    # Group the data by 'Time'
    grouped = frequency_summary.groupby('Time')
    
    for time, group in grouped:
        frequencies = group['Frequency'].values
        total = frequencies.sum()
        probabilities = frequencies / total
        # Calculate the Shannon index
        shannon_index = entropy(probabilities, base=2)
        shannon_indices[time] = shannon_index
        
    return shannon_indices


def plot_shannon_indices(file_paths):
    """
    Plot the Shannon indices over time for multiple datasets.

    Parameters:
    file_paths (list of str): A list of paths to the CSV files to be processed.
    
    Returns:
    The plot of Shannon indices over times.
    """
    plt.figure(figsize=(10, 6))
    
    for file_path in file_paths:
        # Load the data and calculate the Shannon index
        frequency_data = load_and_count_frequency(file_path)[0]
        shannon_indices = calculate_shannon_index(frequency_data)
        
        times = list(shannon_indices.keys())
        indices = list(shannon_indices.values())
        
        label = os.path.splitext(os.path.basename(file_path))[0]
        plt.plot(times, indices, marker='o', label=label)
    
    plt.xlabel('Time')
    plt.ylabel('Shannon Index')
    plt.title('Shannon Index Over Time for Selected Datasets')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to the 'static' directory
    plot_path = 'static/shannon_indices_plot.png' 
    plt.savefig(plot_path)
    plt.close()  


def calculate_pearson_shannon(file_paths):
    """
    Calculate the Pearson correlation between Shannon index and total frequency.

    Parameters:
    file_paths (list of str): A list of paths to the CSV files to be processed.

    Returns:
    dict: A dictionary with file names as keys and Pearson correlation tuples as values.
    """
    correlation = {}
    
    for file_path in file_paths:
        # Load the data and calculate the Shannon index
        frequency_data = load_and_count_frequency(file_path)[0]
        shannon_indices = calculate_shannon_index(frequency_data)
        shannon_indice = pd.DataFrame(list(shannon_indices.items()), columns=['Time', 'Shannon'])
        # Calculate the total frequency for each given time point
        frequency_all_data = load_and_count_frequency(file_path)[1]
        # Calculate the Pearson correlation between Shannon index and total frequency
        Pearson_correlation = pearsonr(shannon_indice['Shannon'], frequency_all_data['Frequency'])
        
        name = os.path.splitext(os.path.basename(file_path))[0]
        correlation[name] = Pearson_correlation
    
    return correlation
    
    
