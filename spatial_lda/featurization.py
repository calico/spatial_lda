import functools
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import KDTree
from scipy.spatial import Voronoi
from tqdm.auto import tqdm


def neighborhood_to_cluster(df, indices):
    clusters = df.iloc[indices].cluster
    return clusters.value_counts()


def neighborhood_to_marker(df, indices, markers):
    return (df.loc[:, markers].iloc[indices] > 0.5).sum()


def neighborhood_to_avg_marker(df, indices, markers):
    return df.loc[:, markers].iloc[indices].mean()


def neighborhood_to_count(df, indices):
    return pd.Series([len(indices)])


def _featurize_cells(df, neighborhood_feature_fn, radius, is_anchor_col,
                     x_col, y_col, z_col=None, include_anchors=False):
    anchor_cells = df[df[is_anchor_col]]
    if include_anchors:
        neighborhood_cells = df
    else:
        neighborhood_cells = df[~df[is_anchor_col]]

    # Throw an error if there are no cells in neighborhoods
    # For example all cells are anchors and include_anchors == False
    if len(neighborhood_cells.index) == 0:
        raise ValueError("There are no neighbours to compute features from \
(try include_anchors = True)")

    feature_fn = functools.partial(neighborhood_feature_fn, neighborhood_cells)
    coord_cols = [x_col, y_col]
    if z_col is not None:
        coord_cols.append(z_col)
    anchor_kdTree = KDTree(anchor_cells[coord_cols].values)
    neighborhood_kdTree = KDTree(neighborhood_cells[coord_cols].values)
    neighborhoods = anchor_kdTree.query_ball_tree(neighborhood_kdTree, radius,
                                                  p=2.0, eps=0.0)
    neighborhood_features = pd.concat(map(feature_fn, neighborhoods), axis=1).T
    neighborhood_features.index = anchor_cells.index
    blank_row_mask = neighborhood_features.isnull().all(axis=1)
    return neighborhood_features[~blank_row_mask]


def _featurize_sample(data, neighborhood_feature_fn, radius, is_anchor_col,
                      x_col, y_col, z_col=None, include_anchors=False):
    i, df = data
    sample_features = _featurize_cells(df, neighborhood_feature_fn, radius,
                                       is_anchor_col, x_col, y_col, z_col=z_col,
                                       include_anchors=include_anchors)
    sample_features.index = map(lambda x: (i, x), sample_features.index)
    return sample_features


def featurize_samples(sample_dfs, neighborhood_feature_fn, radius, is_anchor_col,
                      x_col, y_col, z_col=None, n_processes=None,
                      include_anchors=False):
    """Extract features from a set of cells using aggregate statistics of their local neighborhood.

    Args:
        sample_dfs: Dictionary mapping `sample_id` to sample dataframe. Sample dataframes should contain information
                    about 1 cell per row. Columns should mininmally include x, y (and optionally z) coordinates +
                    a boolean column denoting whether a cell is an index cell or not.
        neighborhood_feature_fn: A function that takes a dataframe containing cells in the neighborhood of an index
                                 cell and returns a single feature vector summarizing the neighborhood. E.g.,
                                 `neighborhood_to_avg_marker` averages the marker intensity for a given marker set.
        radius: l2 distance radius of cells to consider when featurizing cells.
        is_anchor_col: Name of column that is True when a cell is an index cell and False otherwise.
        x_col: Name of column containing x-coordinate.
        y_col: Name of column containing y-coordinate.
        z_col: Name of column containing z-coordinate.
        n_processes: Number of parallel processes to use when featurizing cells.

    Returns:
        One dataframe containing one row per index cell across all samples with their neighborhood features. Index
        cells with empty neighborhoods are removed from this dataframe.
    """


    all_sample_features = []
    featurize_sample_fn = functools.partial(
        _featurize_sample, neighborhood_feature_fn=neighborhood_feature_fn, radius=radius,
        is_anchor_col=is_anchor_col, x_col=x_col, y_col=y_col, z_col=z_col,
        include_anchors=include_anchors)
    if n_processes is not None:
        with Pool(n_processes) as pool:
            total = len(sample_dfs)
            all_sample_features = list(tqdm(pool.imap(featurize_sample_fn,
                                                      sample_dfs.items()),
                                            total=total))
    else:
        for i, sample_df in sample_dfs.items():
            sample_features = featurize_sample_fn((i, sample_df))
            all_sample_features.append(sample_features)

    all_sample_features = pd.concat(all_sample_features).fillna(0)
    return all_sample_features


def featurize_tumors(tumor_dfs, neighborhood_feature_fn, radius=100, n_processes=None):
    return featurize_samples(tumor_dfs, neighborhood_feature_fn, radius, 'is_tumor',
                             'x', 'y', n_processes=n_processes)


def featurize_spleens(spleen_dfs, neighborhood_feature_fn, radius=100, n_processes=None):
    return featurize_samples(spleen_dfs, neighborhood_feature_fn, radius, 'isb',
                             'sample.X', 'sample.Y', z_col='sample.Z',
                             n_processes=n_processes)


def make_nearest_neighbor_graph(sample_features, sample_dfs, sample_idx, x_col, y_col, z_col=None):
    sample_idxs = sample_features.index.map(lambda x: x[0])
    sample_rows = sample_features[sample_idxs == sample_idx]
    cell_idx = sample_rows.index.map(lambda x: x[1])
    if z_col is None:
        coords = [x_col, y_col]
    else:
        coords = [x_col, y_col, z_col]
    cell_coords = sample_dfs[sample_idx].loc[cell_idx][coords].values
    vor = Voronoi(cell_coords)
    num_edges = vor.ridge_points.shape[0]
    num_nodes = len(cell_coords)
    src_nodes = vor.ridge_points[:, 0]
    dst_nodes = vor.ridge_points[:, 1]
    coord_difference = cell_coords[src_nodes] - cell_coords[dst_nodes]
    edge_lengths = np.sqrt(np.sum(coord_difference**2.0, axis=1))
    assert len(edge_lengths) == num_edges
    return num_nodes, src_nodes, dst_nodes, edge_lengths


def make_minimum_spaning_tree_mask(num_nodes, src_nodes, dst_nodes,
                                   edge_lengths):
    num_edges = len(src_nodes)
    adjacency_matrix = coo_matrix((edge_lengths, (src_nodes, dst_nodes)),
                                  shape=(num_nodes, num_nodes)).tocsr()
    edge_index = coo_matrix((np.arange(num_edges), (src_nodes, dst_nodes)),
                            shape=(num_nodes, num_nodes)).tocsr()
    spanning_tree = minimum_spanning_tree(adjacency_matrix)
    mst_mask = np.asarray(edge_index[spanning_tree.nonzero()]).squeeze()
    return mst_mask


def make_difference_matrix(num_nodes, src_nodes, dst_nodes):
    num_edges = len(src_nodes)
    rows = np.hstack([np.arange(num_edges), np.arange(num_edges)])
    cols = np.hstack([src_nodes, dst_nodes])
    values = np.hstack([np.ones(num_edges), -1 * np.ones(num_edges)])
    difference_matrix = coo_matrix(
        (values, (rows, cols)), shape=(num_edges, num_nodes)).tocsr()
    return difference_matrix


def make_merged_difference_matrices(sample_features, sample_dfs,
                                    x_col, y_col, z_col=None,
                                    reduce_to_mst=True):
    difference_matrices = dict()
    sample_idxs = sample_features.index.map(lambda x: x[0])
    for sample_idx in set(sample_idxs):
        graph = make_nearest_neighbor_graph(
            sample_features, sample_dfs, sample_idx, x_col, y_col, z_col=z_col)
        num_nodes, src_nodes, dst_nodes, edge_lengths = graph
        difference_matrix = make_difference_matrix(
            num_nodes, src_nodes, dst_nodes)
        if reduce_to_mst:
            mst_mask = make_minimum_spaning_tree_mask(
                num_nodes, src_nodes, dst_nodes, edge_lengths)
            difference_matrix = difference_matrix[mst_mask, :]
        difference_matrices[sample_idx] = difference_matrix
    return difference_matrices
