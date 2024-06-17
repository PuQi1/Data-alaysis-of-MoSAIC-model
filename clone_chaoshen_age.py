import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os

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


def calculate_chaoshen_index_age(df, time):
    """
    Calculate the Chao-Shen entropy for each age point at a specific time.

    Parameters:
    df (DataFrame): DataFrame with 'Time', 'Clone ID', 'age_in_days', and 'Frequency' columns.
    times (int): Specific time point to calculate the entropy for.

    Returns:
    DataFrame: A DataFrame with 'age_in_days', 'ChaoShen_Index', and 'Dataset' columns.
    """
    # Calculate the sum of frequencies for each age point and clone ID at a given time point
    freq_sum = df[df['Time'] == time].groupby(['age_in_days', 'Clone ID']).agg({'Frequency': 'sum'}).reset_index()
    # Get unique age points
    age_points = freq_sum['age_in_days'].unique()
    results = {
        'age_in_days': age_points,
        # Calculate the Chao-Shen entropy for each age point
        'ChaoShen_Index': [entropy_chao_shen(freq_sum[freq_sum['age_in_days'] == age]['Frequency']) for age in age_points],
        'Dataset': f'{int(time / 30)} months'
    }
    return pd.DataFrame(results)



def plot_chaoshen_index_age(file_path, times):
    """
    Plot the Chao-Shen indices for multiple age points over selected times.

    Parameters:
    file_path (str): The path to the CSV file to be processed.
    times (list of int): List of specific time points to calculate and plot the entropy for.
    
    Returns:
    The plot of Chao-Shen indices over age for selected times.
    """
    # Process data using the calculate_age function
    df = calculate_age(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    # Calculate the Chao-Shen index for each age point at several selected times
    all_data_slow = pd.concat([calculate_chaoshen_index_age(df, t) for t in times])
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=all_data_slow, x='age_in_days', y='ChaoShen_Index', hue='Dataset', marker="o",
                 palette=['grey', 'lightblue', 'lightpink', 'lightgreen'])
    plt.title(f'Chaoshen Index Over Age for {file_name} at Selected Times')
    plt.xlabel('Age in Days')
    plt.ylabel('ChaoShen Index')
    plt.legend(title='Time (months)')
    
    plt.grid(True)
    
    # Save the plot to the 'static' directory
    plot_path = 'static/chaoshen_age_plot.png' 
    plt.savefig(plot_path)
    plt.close()  

    
    
def calculate_pearson_chaoshen_age(file_path, times):
    """
    Calculate the Pearson correlation between Chao-Shen entropy and total frequency for each age point at selected times.

    Parameters:
    file_path (str): The path to the CSV file to be processed.
    times (list of int): List of specific time points to calculate the correlation for.

    Returns:
    dict: A dictionary with time points as keys and Pearson correlation tuples as values.
    """
    # Process data using the calculate_age function
    dataset = calculate_age(file_path)
    # Calculate the pearson correlation coefficients of Chao-Shen index and total number of cells at selected times
    corr_coefficient = {}
    for time in times:
        chaoshen_index = calculate_chaoshen_index_age(dataset, time)
        # Calculate the sum of frequencies for clone ID at each age point at a given time point
        freq_data = dataset[dataset['Time'] == time]
        frequency_sum_age = freq_data.groupby('age_in_days')['Frequency'].sum().reset_index(name='Total_Frequency')

        merged_data = pd.merge(chaoshen_index, frequency_sum_age, on='age_in_days')
        # Calculate the Pearson correlation coefficient
        correlation = pearsonr(merged_data['ChaoShen_Index'], merged_data['Total_Frequency'])
        corr_coefficient[time] = correlation
    
    return corr_coefficient


