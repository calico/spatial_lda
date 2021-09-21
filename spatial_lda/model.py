from collections import OrderedDict
import itertools
import logging
from multiprocessing import Pool
import time
import numpy as np
import pandas as pd
from scipy.special import digamma
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

import spatial_lda.admm as admm
from spatial_lda.online_lda import LatentDirichletAllocation


def _update_xi(counts, diff_matrix, diff_penalty, sample_id, verbosity=0,
               rho=1e-1, mu=2.0, primal_tol=1e-3, threshold=None):
    if verbosity >= 1:
        logging.info(f'>>> Infering topic weights for sample {sample_id}')
    weight = 1. / diff_penalty
    cs = digamma(counts) - digamma(np.sum(counts, axis=1, keepdims=True))
    s = weight * np.ones(diff_matrix.shape[0])
    result = admm.admm(cs, diff_matrix, s, rho, verbosity=verbosity, mu=mu, primal_tol=primal_tol,
                       threshold=threshold)
    if verbosity >= 1:
        logging.info(f'>>> Done inferring topic weights for sample {sample_id}')
    return result


def _wrap_update_xi(inputs):
    return _update_xi(**inputs)


def _update_xis(sample_features, difference_matrices, difference_penalty, gamma,
                n_parallel_processes, verbosity, primal_dual_mu=2, admm_rho=0.1,
                primal_tol=1e-3, threshold=None):
    sample_idxs = sample_features.index.map(lambda x: x[0])
    new_xis = np.zeros_like(gamma)
    if n_parallel_processes > 1:
        with Pool(n_parallel_processes) as pool:
            unique_idxs = np.unique(sample_idxs)
            sample_masks = [
                sample_idxs == sample_idx for sample_idx in unique_idxs]
            sample_counts = [gamma[sample_mask, :]
                             for sample_mask in sample_masks]
            sample_diff_matrices = [difference_matrices[sample_idx]
                                    for sample_idx in unique_idxs]
            diff_penalties = [difference_penalty for _ in unique_idxs]
            tasks = OrderedDict((('counts', sample_counts),
                                 ('diff_matrix', sample_diff_matrices),
                                 ('diff_penalty', diff_penalties),
                                 ('sample_id', unique_idxs),
                                 # Logging causes multiprocessing to get stuck
                                 # (https://pythonspeed.com/articles/python-multiprocessing/)
                                 ('verbosity', itertools.repeat(0)),                               
                                 ('rho', itertools.repeat(admm_rho)),
                                 ('mu', itertools.repeat(primal_dual_mu)),
                                 ('primal_tol', itertools.repeat(primal_tol)),
                                 ('threshold', itertools.repeat(threshold))))
            # convert into a list of keyword dictionaries
            kw_tasks = [{k: v for k, v in zip(tasks.keys(), values)}
                        for values in list(zip(*tasks.values()))]
            results = list(tqdm(pool.imap(_wrap_update_xi, kw_tasks),
                                total=len(unique_idxs),
                                position=1,
                                desc='Update xi'))
            new_xis = np.concatenate(results)
    else:
        for sample_idx in np.unique(sample_idxs):
            sample_mask = sample_idxs == sample_idx
            sample_counts = gamma[sample_mask, :]
            sample_diff_matrix = difference_matrices[sample_idx]
            new_xis[sample_mask] = _update_xi(sample_counts,
                                              sample_diff_matrix,
                                              difference_penalty,
                                              sample_idx,
                                              verbosity=verbosity,
                                              rho=admm_rho,
                                              mu=primal_dual_mu,
                                              primal_tol=primal_tol,
                                              threshold=threshold)
    return new_xis


def train(sample_features, difference_matrices, n_topics, difference_penalty=1,
          n_iters=3, n_parallel_processes=1, verbosity=0,
          primal_dual_mu=2, admm_rho=1.0, primal_tol=1e-3, threshold=None):
    """Train a Spatial-LDA model.
    
    Args:
        sample_features: Dataframe that contains neighborhood features of index cells indexed by (sample ID, cell ID).
                         (See featurization.featurize_samples).
        difference_matrices: Difference matrix corresponding to the spatial regularization structure imposed on the
                             samples. (I.e., which cells should be regularized to have similar priors on topics).
                             (See featurization.make_merged_difference_matrices).
        n_topics: Number of topics to fit.
        difference_penalty: Penalty on topic priors of "adjacent" index cells.
        n_iters: Number of outer-loop iterations (LDA + ADMM) to run.
        n_parallel_processes: Number of parallel processes to use.
        verbosity: Amount of debug / info updates to see.
        primal_dual_mu: mu used in primal-dual updates (see paper for more details).
        admm_rho: rho used in ADMM optimization (see paper for more details).
        primal_tol: tolerance level for primal-dual updates.  In general, this value should not be
                    greater than 0.05.
        threshold: Cutoff for the percent change in the admm objective function.  Must be
                    greater than 0 and less than 1.  Typical value is 0.01.

    Returns:
        A Spatial-LDA model.
    """
    start_time = time.time()
    xis = None
    for i in range(n_iters):
        logging.info(f'>>> Starting iteration {i}')
        m_step_start_time = time.time()
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=0,
                                        n_jobs=n_parallel_processes, max_iter=5,
                                        doc_topic_prior=xis)
        lda.fit(sample_features.values)
        gamma = lda._unnormalized_transform(sample_features.values)
        m_duration = time.time() - m_step_start_time
        logging.info(f'>>> Iteration {i}, M-step took {m_duration} seconds.')
        e_step_start_time = time.time()
        xis = _update_xis(sample_features=sample_features,
                          difference_matrices=difference_matrices,
                          difference_penalty=difference_penalty,
                          gamma=gamma,
                          n_parallel_processes=n_parallel_processes,
                          verbosity=verbosity,
                          primal_dual_mu=primal_dual_mu,
                          admm_rho=admm_rho,
                          primal_tol=primal_tol,
                          threshold=threshold)
        e_duration = time.time() - e_step_start_time
        logging.info(f'>>> Iteration {i}, E-step took {e_duration} seconds.')

    last_m_step_start = time.time()
    columns = ['Topic-%d' % i for i in range(n_topics)]
    lda.topic_weights = pd.DataFrame(lda.fit_transform(sample_features.values),
                                     index=sample_features.index,
                                     columns=columns)
    logging.info(f'>>> Final M-step took {time.time() - last_m_step_start} seconds.')
    logging.info(f'>>> Training took {time.time() - start_time} seconds.')
    return lda


def _topic_name(i):
    return f'Topic-{i}'


def infer(components, sample_features, difference_matrices, difference_penalty=1,
          n_parallel_processes=1):
    """Run inferrence on a Spatial-LDA model.

    This runs only the ADMM updates to get spatially-regularized topic weights on `sample_features`, allowing
    us to infer topic weights on samples after training a model to get topic parameters.
    
    Args:
        components: The components of the Spatial-LDA model (typically `spatial_lda_model.components_`.
        sample_features: Dataframe that contains neighborhood features of index cells indexed by (sample ID, cell ID).
                         (See featurization.featurize_samples).
        difference_matrices: Difference matrix corresponding to the spatial regularization structure imposed on the
                             samples. (I.e., which cells should be regularized to have similar priors on topics).
                             (See featurization.make_merged_difference_matrices).
        difference_penalty: Penalty on topic priors of "adjacent" index cells.
        n_parallel_processes: Number of parallel processes to use.

    Returns:
        Spatial-LDA model with the same topic parameters as the original model but with new topic-weights corresponding
        to the provided sample_features and difference_matrices.
    """
    start_time = time.time()
    logging.info('>>> Starting inference')
    n_topics = components.shape[0]
    complete_lda = LatentDirichletAllocation(n_components=n_topics,
                                             random_state=0,
                                             n_jobs=n_parallel_processes,
                                             max_iter=2,
                                             doc_topic_prior=None)
    complete_lda.set_components(components)
    gamma = complete_lda._unnormalized_transform(sample_features.values)
    xis = _update_xis(sample_features,
                      difference_matrices,
                      difference_penalty,
                      gamma=gamma,
                      n_parallel_processes=n_parallel_processes,
                      verbosity=0)
    complete_lda.doc_topic_prior_ = xis
    columns = [_topic_name(i) for i in range(n_topics)]
    topic_weights = pd.DataFrame(complete_lda.transform(sample_features.values),
                                 index=sample_features.index,
                                 columns=columns)
    complete_lda.topic_weights = topic_weights
    logging.info(f'>>> Inference took {time.time() - start_time} seconds.')
    return complete_lda


def get_component_mapping(stats_1, stats_2):
    similarity = stats_1 @ stats_2.T
    assignment = linear_sum_assignment(-similarity)
    mapping = {k: v for k, v in zip(*assignment)}
    return mapping


def get_consistent_orders(stats_list):
    d = stats_list[0].shape[1]
    n_topics = [stats.shape[0] for stats in stats_list]
    assert all([stats.shape[1] == d for stats in stats_list])
    assert all([n1 <= n2 for n1, n2 in zip(n_topics[:-1], n_topics[1:])])
    orders = [list(range(n_topics[0]))]
    for stats_1, stats_2 in zip(stats_list[:-1], stats_list[1:]):
        n_topics_1 = stats_1.shape[0]
        n_topics_2 = stats_2.shape[0]
        mapping = get_component_mapping(stats_1[orders[-1], :], stats_2)
        mapped = mapping.values()
        unmapped = set(range(n_topics_2)).difference(mapped)
        order = [mapping[k] for k in range(n_topics_1)] + list(unmapped)
        orders.append(order)
    return orders


def apply_order_to_model(model, order):
    model.components_ = model.components_[order, :]
    if not np.isscalar(model.doc_topic_prior):
        model.doc_topic_prior = model.doc_topic_prior[:, order]
    model.doc_topic_prior_ = model.doc_topic_prior_[:, order]
    mapper = {_topic_name(j): _topic_name(i) for i, j in enumerate(order)}
    _df = model.topic_weights
    _df.rename(columns=mapper, inplace=True)
    model.topic_weights = _df.reindex(sorted(_df.columns), axis=1)


def order_topics_consistently(models, use_topic_weights=True):
    if use_topic_weights:
        stats_list = [model.topic_weights.values.T for model in models]
    else:
        stats_list = [model.components_ for model in models]

    orders = get_consistent_orders(stats_list)
    for model, order in zip(models, orders):
        apply_order_to_model(model, order)
