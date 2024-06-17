from flask import Flask, render_template, request, session, redirect, url_for
from clone_overlapping import plot_common_clones_over_time
from clone_shannon import plot_shannon_indices, calculate_pearson_shannon
from clone_chaoshen import plot_chaoshen_indices, calculate_pearson_chaoshen
from clone_chaoshen_age import plot_chaoshen_index_age, calculate_pearson_chaoshen_age
from clone_shannon_age import plot_shannon_index_age, calculate_pearson_shannon_age
from clone_cluster_dataset import plot_scatter_nonzero_ids, plot_scatter_frequencies, analyze_clusters, tsne_visualization
from clone_cluster import cluster_for_top_n, tsne_visualize_for_top_n
from clone_cluster_age import cluster_for_top_n_age, tsne_visualize_for_top_n_age
import os

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def overlap():
    if request.method == 'POST':
        selected_datasets = request.form.getlist('dataset')
        data_dir = 'data'
        dataset_files = {
            'memfast': 'memfast.csv',
            'memslow': 'memslow.csv',
            'naidis': 'naidis.csv',
            'naiinc': 'naiinc.csv'
        }
        file_paths = [os.path.join(data_dir, dataset_files[dataset]) for dataset in selected_datasets]

        plot_common_clones_over_time(file_paths)
        return redirect(url_for('overlap'))

    return render_template('overlap.html')

@app.route('/shannon', methods=['GET', 'POST'])
def shannon():
    if request.method == 'POST':
        selected_datasets = request.form.getlist('dataset')
        data_dir = 'data'
        dataset_files = selected_datasets  # directly using the list from form

        full_paths = [os.path.join(data_dir, file) for file in dataset_files]
        plot_shannon_indices(full_paths)
        correlations = calculate_pearson_shannon(full_paths)
        return render_template('shannon.html', correlations=correlations)

    return render_template('shannon.html')


@app.route('/chaoshen', methods=['GET', 'POST'])
def chaoshen():
    if request.method == 'POST':
        selected_datasets = request.form.getlist('dataset')
        data_dir = 'data'
        dataset_files = selected_datasets  # directly using the list from form

        full_paths = [os.path.join(data_dir, file) for file in dataset_files]
        plot_chaoshen_indices(full_paths)
        correlations = calculate_pearson_chaoshen(full_paths)
        return render_template('chaoshen.html', correlations=correlations)

    return render_template('chaoshen.html')


@app.route('/chaoshen_age', methods=['GET', 'POST'])
def chaoshen_age():
    if request.method == 'POST':
        selected_file = request.form['file']
        selected_times = request.form.getlist('times')
        times = list(map(int, selected_times))
        data_dir = 'data'
        file_path = os.path.join(data_dir, selected_file)

        plot_chaoshen_index_age(file_path, times)
        correlations = calculate_pearson_chaoshen_age(file_path, times)

        return render_template('chaoshen_age.html', correlations=correlations)
    return render_template('chaoshen_age.html')


@app.route('/shannon_age', methods=['GET', 'POST'])
def shannon_age():
    if request.method == 'POST':
        selected_file = request.form['file']
        selected_times = request.form.getlist('times')
        times = list(map(int, selected_times))
        data_dir = 'data'
        file_path = os.path.join(data_dir, selected_file)

        plot_shannon_index_age(file_path, times)
        correlations = calculate_pearson_shannon_age(file_path, times)

        return render_template('shannon_age.html', correlations=correlations)
    return render_template('shannon_age.html')


@app.route('/cluster_dataset', methods=['GET', 'POST'])
def cluster_dataset():
    if request.method == 'POST':
        action = request.form.get('action')
        file_paths = request.form.getlist('file_paths')
        time_value = int(request.form.get('time_value', 0))
        

        if action == 'analyze_basic':
            # Generate initial plots
            full_paths = [os.path.join('data', path) for path in file_paths]
            plot_urls = {
                'scatter_nonzero': plot_scatter_nonzero_ids(full_paths, time_value),
                'scatter_freq': plot_scatter_frequencies(full_paths, time_value),
                'analyze_clusters': analyze_clusters(full_paths, time_value)
            }
            return render_template('cluster_dataset.html', plot_urls=plot_urls)

        elif action == 'analyze_clusters':
            # Generate the t-SNE visualization
            n_clusters = int(request.form.get('n_clusters', 2))
            full_paths = [os.path.join('data', path) for path in file_paths]
            tsne_plot = tsne_visualization(full_paths, time_value, n_clusters)
            print(f"Generated t-SNE plot: {tsne_plot}")
            return render_template('cluster_dataset.html', tsne_plot=tsne_plot)

    return render_template('cluster_dataset.html')


@app.route('/cluster', methods=['GET', 'POST'])
def cluster():
    if request.method == 'POST':
        action = request.form.get('action')
        selected_file = request.form['file']
        num = int(request.form.get('num', 1000))

        if action == 'analyze_basic':
            # Generate initial plots
            full_path = os.path.join('data', selected_file) 
            plot_url = cluster_for_top_n(full_path, num)
            return render_template('cluster.html', plot_urls=plot_url)

        elif action == 'analyze_clusters':
            # Generate the t-SNE visualization
            n_clusters = int(request.form.get('n_clusters', 2))
            full_path = os.path.join('data', selected_file) 
            tsne_plot = tsne_visualize_for_top_n(full_path, num, n_clusters)
            print(f"Generated t-SNE plot: {tsne_plot}")
            return render_template('cluster.html', tsne_plot=tsne_plot)

    return render_template('cluster.html')


@app.route('/cluster_age', methods=['GET', 'POST'])
def cluster_age():
    if request.method == 'POST':
        action = request.form.get('action')
        selected_file = request.form['file']
        num = int(request.form.get('num', 1000))
        time_value = int(request.form.get('time_value', 0))

        if action == 'analyze_basic':
            # Generate initial plots
            full_path = os.path.join('data', selected_file) 
            plot_url = cluster_for_top_n_age(full_path, time_value, num)
            return render_template('cluster_age.html', plot_urls=plot_url)

        elif action == 'analyze_clusters':
            # Generate the t-SNE visualization
            n_clusters = int(request.form.get('n_clusters', 2))
            full_path = os.path.join('data', selected_file) 
            tsne_plot = tsne_visualize_for_top_n_age(full_path, time_value, num, n_clusters)
            print(f"Generated t-SNE plot: {tsne_plot}")
            return render_template('cluster_age.html', tsne_plot=tsne_plot)

    return render_template('cluster_age.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


