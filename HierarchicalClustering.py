'''
HierarchicalClustering.py

Holds the Hierarchical Clustering Functionality as learned from "Advances in Financial Machine Learning"
by Marcos Lopes de Prado

Public function:
    * hierarchical_cluster(data)
    * connected_clusters(organized_corr_matrix, correlation_threshol)

'''

# Hierarchical clustering
from itertools import combinations
import numpy as np
from itertools import product, combinations_with_replacement
import pandas as pd


class Cluster:
    def __init__(self):
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def get_end_child(self):
        if len(self.children) == 1:
            return self.children
        else:
            l_1 = self.children[0].get_end_child()
            l_2 = self.children[1].get_end_child()
            return l_1 + l_2


def maximum_linkage(cluster1, cluster2, distances):
    link_distances = []
    for cluster_1_child in cluster1.get_end_child():
        for cluster_2_child in cluster2.get_end_child():
            try:
                link_distances.append(distances[cluster_1_child + cluster_2_child])

            except:
                link_distances.append(distances[cluster_2_child + cluster_1_child])

    return max(link_distances)


def minimal_cluster_pair(set_clusters, distances, linkage_function):
    smallest_score = np.inf
    smallest_set_clusters = []

    for pair in combinations(set_clusters, 2):
        current_score = linkage_function(pair[0], pair[1], distances)

        if current_score < smallest_score:
            smallest_score = current_score
            smallest_set_clusters = pair

    return smallest_set_clusters


def create_cluster(sub_cluster1, sub_cluster2):
    new_cluster = Cluster()
    new_cluster.add_child(sub_cluster1)
    new_cluster.add_child(sub_cluster2)
    return new_cluster


def update_set_clusters(list_sub_clusters, set_clusters):
    # Create the new cluster
    cluster = create_cluster(list_sub_clusters[0], list_sub_clusters[1])

    # remove the sub clusters from the set_clusters container
    set_clusters.remove(list_sub_clusters[0])
    set_clusters.remove(list_sub_clusters[1])

    # Add the new cluster to the set_clusters container
    set_clusters.append(cluster)

    return set_clusters


def get_hierarchical_weights(set_clusters, distances, weights):
    while len(set_clusters) > 1:
        min_clusters = minimal_cluster_pair(set_clusters, distances, maximum_linkage)
        nodes_to_change_weights = min_clusters[0].get_end_child() + min_clusters[1].get_end_child()
        for node in nodes_to_change_weights:
            weights[node] = 0.5 * weights[node]
        set_clusters = update_set_clusters(min_clusters, set_clusters)
    return set_clusters[0], weights


def distance(corr):
    try:
        return 2 * np.sqrt(1 - corr)
    except:
        return np.inf


def get_edge_lengths(distance_matrix):
    distances = {}
    for i, j in combinations(list(distance_matrix.columns), 2):
        try:
            value = distance_matrix[i].loc[j]
            distances[i + j] = value
        except Exception as e:
            print(str(e))
    return distances


def initialize_set_clusters(list_tickers):
    set_clusters = []
    for ticker in list_tickers:
        cluster = Cluster()
        cluster.add_child(ticker)
        set_clusters.append(cluster)
    return set_clusters


def hierarchical_cluster(data):
    '''
    Clustered correlation matrix for a data set.
    :param data: A pandas data frame
    :return: Organized correlation data frame
    '''
    corr_matrix = data.corr()

    distance_matrix = corr_matrix.applymap(distance)
    distances = get_edge_lengths(distance_matrix)

    set_clusters = initialize_set_clusters(list(distance_matrix.columns))

    weights = {i: 1 for i in data.columns}

    cluster, optimal_weights = get_hierarchical_weights(set_clusters, distances, weights)

    clustered_index = cluster.get_end_child()

    return corr_matrix[clustered_index].loc[clustered_index]


def connected_clusters(organized_corr_matrix, correlation_threshold=0.5):
    '''
    Gives a list of connected columns for an organized heat map.
    :param organized_corr_matrix: hierarchically clustered data frame
    :param correlation_threshold: threshold for minimum correlation values
    :return: a list of set of connected columns
    '''
    # Note: not efficient, just used to get the functionality working properly.
    # I am sure that it can be made better via graph theory... (later project)
    dict_corr_vals = {}

    # Get all pairs of columns and relate the correlation value to the pair
    pairs = combinations_with_replacement(organized_corr_matrix.columns, 2)
    for i, p in enumerate(pairs):
        p0, p1 = p[0], p[1]
        if p0 != p1:
            val = organized_corr_matrix[p0].loc[p1]
            dict_corr_vals[i] = [val, p]

    corr_val_df = pd.DataFrame(dict_corr_vals, index=["value", "pair"]).T
    corr_val_df.sort_values("value", inplace=True)

    # Filter the correlation values by the threshold
    top_corr_df = corr_val_df.loc[abs(corr_val_df.value) > correlation_threshold]
    pairs = list(top_corr_df.pair)

    # Initialize the current_cluster
    current_cluster = None

    max_iteration = 50
    for _ in range(max_iteration):
        for item in pairs:
            if current_cluster is None:
                current_cluster = set()
                current_cluster.update(item)
                pairs.remove(item)
            else:
                if any(x in item for x in current_cluster):
                    current_cluster.update(item)
                    pairs.remove(item)

        pairs.append(tuple(current_cluster))

        current_cluster = None

    return pairs


def _test_(save_plot=False):
    import matplotlib.pyplot as plt
    import seaborn as sns

    font = {'size': 5}
    plt.rc('font', **font)

    # Note: local file
    file = "../KaggleProjects/KaggleDataSets/fifa_data/fifa_data.csv"

    df = pd.read_csv(file, index_col=0)
    df = df[df.dtypes[df.dtypes == "float64"].index]

    df_corr = df.corr()
    df_hier_corr = hierarchical_cluster(df)

    fig, ax = plt.subplots(2, figsize=(8, 10))

    ax[0].set_title("Original Correlation Heatmap")
    sns.heatmap(df_corr, ax=ax[0])

    ax[1].set_title("Hierarchical Correlation Heatmap")
    sns.heatmap(df_hier_corr, ax=ax[1])

    fig.subplots_adjust(hspace=0.5)

    conn_clusters = connected_clusters(df_hier_corr, 0.8)
    print(conn_clusters)

    if save_plot:
        plt.savefig("HierarchicalClustering_fifa_data.png")

    else:
        plt.show()


if __name__ == "__main__":
    _test_()