import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import colors
import palettable.cartocolors.qualitative as qual_palettes
import pandas as pd
import pickle
import scipy
import seaborn as sns

blue = sns.color_palette()[0]
green = sns.color_palette()[1]
red = sns.color_palette()[2]


def make_rgb_channel(data, hue):
    def normalize(arr):
        arr = arr - np.percentile(arr, 5)
        return np.clip(arr / np.percentile(arr, 97), 0, 1)
    color_data = np.zeros(list(data.shape) + [3])
    color_data[..., 0] = hue
    color_data[..., 1] = 1
    color_data[..., 2] = normalize(data)
    return colors.hsv_to_rgb(color_data)


def make_multichannel_im(data, starting_hue=0.):
    num_channels = data.shape[-1]
    data_shape = data.shape[:-1]
    s = []
    for i in range(num_channels):
        hue = (starting_hue + (i * 1. / num_channels))
        if hue > 1:
            hue -= 1
        s.append(make_rgb_channel(data[..., i], hue))
    s = np.stack(s)
    s[s < 1e-5] = np.NAN
    normalized_s = (np.nanmean(s**5, axis=0))**(0.2)
    return normalized_s


def plot_one_tumor_false_color(ax, tumor_idx, topic_weights, patient_dfs):
    mask = topic_weights.index.map(lambda x: x[0]) == tumor_idx
    tumor_topics = topic_weights[mask]
    rgb_points = pd.DataFrame(make_multichannel_im(tumor_topics.values,
                                                   starting_hue=0.15),
                              index=tumor_topics.index)
    cell_coords = patient_dfs[tumor_idx]
    immune_coords = cell_coords[cell_coords.isimmune]
    cell_indices = rgb_points.index.map(lambda x: x[1])
    coords = patient_dfs[tumor_idx].loc[cell_indices]
    ax.scatter(immune_coords['y'], -immune_coords['x'],
               s=5, c='k', label='Immune', alpha=0.1)
    ax.scatter(coords['y'], -coords['x'], s=3, c=rgb_points.values)
    ax.set_title("Tumor %d" % tumor_idx)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)


def plot_one_tumor_cluster(ax, tumor_idx, features, patient_dfs,
                           cluster_colors=[green, red]):
    cell_coords = patient_dfs[tumor_idx]
    immune_coords = cell_coords[cell_coords.isimmune]
    ax.scatter(immune_coords['y'], -immune_coords['x'], s=5, c='k',
               label='Immune', alpha=0.1)
    clusters = features.cluster.unique()
    for i, color in zip(clusters, cluster_colors):
        cluster_cell_indices = features[features.cluster == i].index.map(
            lambda x: x[1])
        coords = patient_dfs[tumor_idx].loc[cluster_cell_indices]
        ax.scatter(coords['y'], -coords['x'], s=3,
                   c=color, label='Cluster %d' % i)
    ax.set_title("Tumor %d" % tumor_idx)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)


def plot_one_tumor_topic(ax, tumor_idx, topic_weights, patient_dfs):
    cell_coords = patient_dfs[tumor_idx]
    immune_coords = cell_coords[cell_coords.isimmune]
    cell_indices = topic_weights.index.map(lambda x: x[1])
    coords = patient_dfs[tumor_idx].loc[cell_indices]
    ax.scatter(immune_coords['y'], -immune_coords['x'],
               s=5, c='k', label='Immune', alpha=0.1)
    ax.scatter(coords['y'], -coords['x'], s=2, c=topic_weights, cmap="Reds",
               label='Topic weight')
    ax.set_title("Tumor %d" % tumor_idx)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)


def plot_one_tumor_all_topics(ax, tumor_idx, topic_weights, patient_dfs):
    color_palette = qual_palettes.Bold_10.mpl_colors
    colors = np.array(color_palette[:topic_weights.shape[1]])
    cell_coords = patient_dfs[tumor_idx]
    immune_coords = cell_coords[cell_coords.isimmune]
    cell_indices = topic_weights.index.map(lambda x: x[1])
    coords = patient_dfs[tumor_idx].loc[cell_indices]
    ax.scatter(immune_coords['y'], -immune_coords['x'],
               s=5, c='k', label='Immune', alpha=0.1)
    ax.scatter(coords['y'], -coords['x'], s=2,
               c=colors[np.argmax(np.array(topic_weights), axis=1), :])
    ax.set_title("Tumor %d" % tumor_idx)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)


def plot_tumors(features_df, plot_fn):
    sns.set_style("white")
    tumor_idx = features_df.index.map(lambda x: x[0])
    tumor_set = np.unique(tumor_idx)
    num_rows = (len(tumor_set) // 4) + 1
    _, axes = plt.subplots(num_rows, 4, figsize=(4 * 4, num_rows * 4))

    for i, tumor in enumerate(tumor_set):
        row = i // 4
        col = i % 4
        plot_fn(axes[row, col], tumor, features_df[tumor_idx == tumor])
    sns.despine(left=True, bottom=True)


def plot_samples_in_a_row(features_df, plot_fn, patient_dfs, tumor_set=None):
    sns.set_style("white")
    tumor_idx = features_df.index.map(lambda x: x[0])
    if tumor_set is None:
        tumor_set = np.unique(tumor_idx)

    n = len(tumor_set)
    num_rows = 1
    num_cols = (len(tumor_set) // num_rows)
    _, axes = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))

    for i, tumor in enumerate(tumor_set):
        plot_fn(axes[i], tumor, features_df[tumor_idx == tumor], patient_dfs)

    sns.despine(left=True, bottom=True)


def plot_bcell_topic_multicolor(ax, sample_idx, topic_weights, spleen_dfs):
    color_palette = qual_palettes.Bold_10.mpl_colors
    colors = np.array(color_palette[:topic_weights.shape[1]])
    cell_coords = spleen_dfs[sample_idx]
    non_b_coords = cell_coords[~cell_coords.isb]
    ax.scatter(
        non_b_coords['sample.Y'],
        non_b_coords['sample.X'],
        s=1,
        c='k',
        marker='x',
        label='Non-B',
        alpha=.2)

    cell_indices = topic_weights.index.map(lambda x: x[1])
    coords = spleen_dfs[sample_idx].loc[cell_indices]

    ax.scatter(coords['sample.Y'], coords['sample.X'], s=3,
               c=colors[np.argmax(np.array(topic_weights), axis=1), :])

    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.set_title(f'Sample {sample_idx}')


def _standardize_topics(topics):
    topics = topics.T
    topics = topics - topics.mean(axis=1, keepdims=True)
    topics = topics / topics.std(axis=1, keepdims=True)
    return topics


def plot_topics_heatmap(topics, features, normalizer=None):
    n_topics = topics.shape[0]
    if normalizer is not None:
        topics = normalizer(topics)
    else:
        topics = _standardize_topics(topics)

    topics = pd.DataFrame(topics, index=features,
                          columns=['Topic %d' % x for x in range(n_topics)])
    sns.heatmap(topics, square=True, cmap='RdBu')


def get_tumor_markers(tumor_features, patient_dfs):
    tumor_idx = tumor_features.index.map(lambda x: x[0])
    tumor_set = np.unique(tumor_idx)
    num_rows = (len(tumor_set) // 4) + 1
    markers = []
    for i, tumor in enumerate(tumor_set):
        cell_idx = tumor_features[tumor_idx == tumor].index.map(lambda x: x[1])
        feature_df = patient_dfs[tumor].loc[cell_idx]
        feature_df = feature_df.set_index(
            tumor_features[tumor_idx == tumor].index)
        markers.append(feature_df)
    markers = pd.concat(markers)
    return markers


def plot_adjacency_graph(ax, tumor_idx, features_df, patient_dfs,
                         difference_matrices):
    sns.set_style("white")
    tumor_idxs = features_df.index.map(lambda x: x[0])
    tumor_rows = features_df[tumor_idxs == tumor_idx]
    cell_idx = tumor_rows.index.map(lambda x: x[1])
    cell_coords = patient_dfs[tumor_idx].loc[cell_idx][['x', 'y']].values
    difference_matrix = difference_matrices[tumor_idx]
    for i in range(difference_matrix.shape[0]):
        src_idx = difference_matrix[i].nonzero()[1][0]
        dst_idx = difference_matrix[i].nonzero()[1][1]
        src_x, src_y = cell_coords[src_idx, :]
        dst_x, dst_y = cell_coords[dst_idx, :]
        ax.plot([src_x, dst_x], [-src_y, -dst_y])

    ax.set_aspect(1)
    ax.set_title(f"Sample {tumor_idx} adjacency")
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
