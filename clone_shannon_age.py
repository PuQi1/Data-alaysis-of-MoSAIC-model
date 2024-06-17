import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from scipy.stats import pearsonr
import os

def calculate_age(file_path):
    """
    Load the CSV file, standardize its column names, and calculate ages of each clone ID in days.

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


def calculate_shannon_index(df):
    """
    Calculate the Shannon index for a given DataFrame.

    Parameters:
    df (DataFrame): The DataFrame containing the frequency data.

    Returns:
    float: The calculated Shannon index.
    """
    probabilities = df['Frequency'] / df['Frequency'].sum()
    return entropy(probabilities, base=2)


def calculate_shannon_index_age(df, time):
    """
    Calculate the Shannon index for each age point at a specific time.

    Parameters:
    df (DataFrame): The DataFrame containing the frequency data.
    time (int): The specific time point to calculate the Shannon index for.

    Returns:
    DataFrame: A DataFrame with 'age_in_days' and 'Shannon_Index' columns.
    """
    freq_data = df[df['Time'] == time]
    # Calculate the sum of frequencies for each age point and clone ID at a given time point
    freq_sum = freq_data.groupby(['age_in_days', 'Clone ID'])['Frequency'].sum().reset_index()
    # Calculate the Shannon index for each age point
    shannon_index = freq_sum.groupby('age_in_days').apply(calculate_shannon_index).reset_index()
    shannon_index.columns = ['age_in_days', 'Shannon_Index']
    shannon_index['Dataset'] = f'{time//30} months'
    
    return pd.DataFrame(shannon_index)
    
    
def plot_shannon_index_age(file_path, times):
    """
    Plot the Shannon index for different age points at selected times.

    Parameters:
    file_path (str): The path to the CSV file to be processed.
    times (list of int): List of specific time points to calculate and plot the Shannon index for.
    
    Returns:
    The plot of Chao-Shen indices over age for selected times.
    """
    # Process data using the calculate_age function
    dataset = calculate_age(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Calculate the Shannon index for each age point at several selected times
    shannon_data = [calculate_shannon_index_age(dataset, time) for time in times]
    all_data = pd.concat(shannon_data)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=all_data, x='age_in_days', y='Shannon_Index', hue='Dataset', palette='tab10', marker='o')
    plt.title(f'Shannon Index Over Age for {file_name} at Selected Times')
    plt.xlabel('Age in Days')
    plt.ylabel('Shannon Index')
    plt.legend(title='Time (months)')
    plt.grid(True)
    
    # Save the plot to the 'static' directory
    plot_path = 'static/shannon_age_plot.png'  # Def
    plt.savefig(plot_path)
    plt.close() 
    

def calculate_pearson_shannon_age(file_path, times):
    """
    Calculate the Pearson correlation between Shannon index and total frequency for each age point at selected times.

    Parameters:
    file_path (str): The path to the CSV file to be processed.
    times (list of int): List of specific time points to calculate the correlation for.

    Returns:
    dict: A dictionary with time points as keys and Pearson correlation tuples as values.
    """
    # Process data using the calculate_age function
    dataset = calculate_age(file_path)
    # Calculate the pearson correlation coefficients of Shannon index and total number of cells at selected times
    corr_coefficient = {}
    for time in times:
        shannon_index = calculate_shannon_index_age(dataset, time)
        # Calculate the sum of frequencies for clone ID at each age point at a given time point
        freq_data = dataset[dataset['Time'] == time]
        frequency_sum_age = freq_data.groupby('age_in_days')['Frequency'].sum().reset_index(name='Total_Frequency')

        merged_data = pd.merge(shannon_index, frequency_sum_age, on='age_in_days')
        # Calculate the Pearson correlation coefficient
        correlation = pearsonr(merged_data['Shannon_Index'], merged_data['Total_Frequency'])
        corr_coefficient[time] = correlation
    
    return corr_coefficient



