import logging
import pickle
import time

import numpy as np
from scipy.special import gammaln, digamma, polygamma
import scipy.sparse
import scipy.sparse.linalg

ALPHA = 0.1
BETA = 0.5
MAXLSITER = 60
MAXITER = 100
#TOL = 1e-3
ADMM_RESIDUAL_RATIO_BOUND = 4.0
ADMM_RHO_SCALE = 2.0


def split_gamma(gamma, n, k, l):
    xi = np.reshape(gamma[0:n * k], [n, k])
    chi = np.reshape(gamma[n * k: n * k + l * k], [l, k])
    return xi, chi


def make_A(D, k):
    return scipy.sparse.kron(D, np.eye(k))


def make_C(D, k):
    """Given differencing matrix on samples D construct C."""
    l = D.shape[0]
    A = make_A(D, k)
    I = scipy.sparse.eye(l * k)
    As = scipy.sparse.vstack((A, -A))
    Is = scipy.sparse.vstack((-I, -I))
    return scipy.sparse.hstack((As, Is))


def f0(gamma, e, rho, s):
    """Compute f0 for primal vars xi and chi, under gaussian appearance with
    mean e, std 1/rho, and edge weights s."""
    # n documents, k topics, l ties between documents
    n, k = e.shape
    l = s.shape[0]
    xi, chi = split_gamma(gamma, n, k, l)
    objective = rho / 2 * np.sum((xi - e)**2.0)
    objective += np.sum(scipy.sparse.diags(s, 0).dot(chi))
    return objective


def gradient_f0(gamma, e, rho, s):
    """Compute gradient of objective given variable gamma, counts c, and edge weights s."""
    n, k = e.shape
    l = s.shape[0]
    xi, chi = split_gamma(gamma, n, k, l)
    gxi = rho * (xi - e)
    gchi = scipy.sparse.diags(s, 0).dot(np.ones((l, k)))
    gxi = np.reshape(gxi, (n * k, 1))
    gchi = np.reshape(gchi, (l * k, 1))
    return np.vstack((gxi, gchi))


def hessian_f0(gamma, e, rho, s):
    """Compute Hessian of objective given xi and count of edges"""
    n, k = e.shape
    l = s.shape[0]
    xi, chi = split_gamma(gamma, n, k, l)
    nabla2_xi = rho * scipy.sparse.eye(n * k)
    zeros_nk_lk = scipy.sparse.coo_matrix((n * k, l * k))
    zeros_lk_lk = scipy.sparse.coo_matrix((l * k, l * k))
    H = scipy.sparse.vstack((scipy.sparse.hstack((nabla2_xi, zeros_nk_lk)),
                             scipy.sparse.hstack((zeros_nk_lk.T, zeros_lk_lk))))
    return H


def r_dual(gamma, u, C, e, rho, s):
    g = gradient_f0(gamma, e, rho, s)
    r = np.squeeze(g) + np.squeeze((C.T.dot(u)))
    return r


def r_cent(gamma, u, C, t):
    f1 = C.dot(gamma)
    return -np.squeeze(scipy.sparse.diags(u, 0).dot(f1)) - 1. / t


def compute_r(gamma, u, C, e, rho, s, t):
    r1 = r_dual(gamma, u, C, e, rho, s)
    r2 = r_cent(gamma, u, C, t)
    r = -np.hstack((r1, r2))
    return r


def build_linear_system(gamma, u, C, e, rho, s, t):
    """Build the linear system for ADMM.primal_dual (see appendix section 5.2.7)."""
    n, k = e.shape
    H = hessian_f0(gamma, e, rho, s)
    uC = scipy.sparse.diags(np.squeeze(u), 0).dot(C)
    Cg = scipy.sparse.diags(np.squeeze(C.dot(gamma)))
    M = scipy.sparse.vstack((scipy.sparse.hstack((H, C.T)),
                             scipy.sparse.hstack((-uC, -Cg)))).tocsr()
    r = compute_r(gamma, u, C, e, rho, s, t)
    return M, r


def split_primal_dual_vars(z, n, k, l):
    gamma = z[:n * k + l * k]
    u = z[n * k + l * k:]
    return np.squeeze(gamma), np.squeeze(u)


def gap(gamma, C, u):
    return -np.sum(C.dot(np.squeeze(gamma)) * np.squeeze(u))


def spsolve(M, r):
    perm = scipy.sparse.csgraph.reverse_cuthill_mckee(M, True)
    inv_perm = np.argsort(perm)
    M = M[perm, :][:, perm]
    r = r[perm]
    delta = scipy.sparse.linalg.spsolve(M, r)
    return delta[inv_perm]


def line_search(gamma, u, C, e, rho, s, t, l):
    """Line search for ADMM.primal_dual (see appendix section 5.2.4)."""
    n, k = e.shape
    M, r = build_linear_system(gamma, u, C, e, rho, s, t)
    delta = spsolve(M, r)
    dgamma, du = split_primal_dual_vars(delta, n, k, l)
    step_max = 1.0
    neg_du = du < 0
    if np.any(neg_du):
        step_max = np.min((step_max, np.min(u[neg_du] / (-du[neg_du]))))

    step = step_max * 0.99
    for lsit in range(MAXLSITER):
        new_gamma = gamma + step * dgamma
        new_u = u + step * du
        new_r = compute_r(new_gamma, new_u, C, e, rho, s, t)
        if (np.any(C.dot(new_gamma) > 0) or
                np.linalg.norm(new_r) > (1 - ALPHA * step) * np.linalg.norm(r)):
            step = step * BETA
        else:
            u = new_u
            gamma = new_gamma
            r = new_r
            break
    if lsit == MAXLSITER - 1:
        logging.warning('Line search failed.')
    return gamma, u, step, lsit


def primal_dual(e, rho, D, s, mu=2, verbosity=0, max_iter=MAXITER, primal_tol=1e-3):
    """ADMM.primal_dual for fusion problem (see appendix section 5.2.4)."""
    l, n = D.shape
    _, k = e.shape
    assert e.shape[0] == D.shape[1]
    gamma = np.hstack((np.ones(n * k), np.ones(l * k)))
    u = np.ones(2 * l * k)
    C = make_C(D, k)
    t = 1.0
    for it in range(max_iter):
        nu = gap(gamma, C, u)
        t = np.max((2 * mu / nu, t * 1.2))
        gamma, u, step, lsit = line_search(gamma, u, C, e, rho, s, t, l)
        r = np.linalg.norm(r_dual(gamma, u, C, e, rho, s)) + \
            np.linalg.norm(r_cent(gamma, u, C, t))
        if verbosity >= 3:
            logging.info(
                f'\tPrimal Dual it: {it}, gap: {nu:.6g}, t: {t:.6g}, step: {step:.6g}, res: {r:.6g}, lsit: {lsit}')
        xis, chis = split_gamma(gamma, n, k, l)
        if r < primal_tol and nu < primal_tol:
            break
    if verbosity >= 2:
        logging.info(
            f'\tPrimal Dual it: {it}, gap: {nu:.6g}, t: {t:.6g}, step: {step:.6g}, res: {r:.6g}, lsit: {lsit}')

    if it == max_iter - 1:
        logging.warn('\tPrimal dual did not converge.')
        with open('pd.dbg.pkl', 'wb') as f:
            pickle.dump((e, rho, D, s, mu), f)
        raise Exception('Stopping in admm.primal_dual')

    return gamma, u


def li(taus, r, rho):
    """Compute negative regularized dirichlet log-likelihood"""
    n, k = r.shape
    r = np.reshape(r, (n, k))
    taus = np.reshape(taus, (n, k))
    objective = np.sum(gammaln(taus))
    objective -= np.sum(gammaln(np.sum(taus, axis=1)))
    objective += rho / 2 * np.sum((taus - r)**2.0)
    return objective


def gradient_li(taus, r, rho):
    n, k = r.shape
    r = np.reshape(r, (n, k))
    taus = np.reshape(taus, (n, k))
    gtau = (digamma(taus) - digamma(np.sum(taus, axis=1, keepdims=True)))
    gtau += rho * (taus - r)
    return np.reshape(gtau, (-1, 1))


def assemble_block_diag(mats):
    row = []
    col = []
    data = []
    offset = 0
    nrows, ncols = mats[0].shape
    row_idx, col_idx = np.meshgrid(range(nrows), range(ncols))
    for a in mats:
        row.append((row_idx + offset).flatten())
        col.append((col_idx + offset).flatten())
        data.append(a.flatten())
        offset += nrows
    data = np.hstack(data)
    row = np.hstack(row)
    col = np.hstack(col)
    return scipy.sparse.coo_matrix((data, (row, col)))


def hessian_li(taus, r, rho):
    n, k = r.shape
    taus = np.reshape(taus, [n, k])
    blocks = []
    for i in range(n):
        block = np.diag(
            polygamma(1, taus[i, :]) + rho) - polygamma(1, np.sum(taus[i, :]))
        blocks.append(block)
    H = assemble_block_diag(blocks)
    return H


def get_update_step(taus, r, rho):
    n, k = r.shape
    g = gradient_li(taus, r, rho)
    taus = np.reshape(taus, [n, k])
    q = polygamma(1, taus) + rho
    z = -polygamma(1, np.sum(taus, axis=1, keepdims=True))
    iq = np.reshape(1 / q, [n, k])
    g = np.reshape(g, [n, k])
    b = np.sum(g * iq, axis=1, keepdims=True) / \
        (1 / z + np.sum(iq, axis=1, keepdims=True))
    step = (g - b) * iq
    sc = np.sum(step * g)
    return np.reshape(step, [-1, 1]), sc, g


def newton_regularized_dirichlet(
        rho, r, max_iter=30, ls_iter=10, tol=1e-4, verbose=False, alpha=0.01, beta=0.5, verbosity=0):
    """Newton optimization for the regularized Dirichlet step of ADMM (see appendix section 5.2.8)."""
    n, k = r.shape
    taus = np.ones((np.prod(r.shape), 1))
    new_li = li(taus, r, rho)
    for it in range(max_iter):
        old_li = new_li
        step, sc, g = get_update_step(taus, r, rho)
        g_norm = np.linalg.norm(g)
        t = 1.0
        neg = (taus - step) < 0
        if len(taus[neg]) > 0:
            t = np.min([t, 0.99 * np.min(taus[neg] / step[neg])])

        for ls_it in range(ls_iter):
            new_taus = taus - t * step
            new_li = li(new_taus, r, rho)
            if verbosity >= 3:
                logging.info(f'  Line search: {ls_it} neg.log.lik.: {new_li}'
                             f' old neg.log.lik.:{old_li}'
                             f' sc:{sc} t:{t}')
            # Armijo-Goldstein condition
            if new_li > old_li - t * alpha * sc:
                t = t * beta
            else:
                taus = new_taus
                break

        if verbosity >= 3:
            logging.info(
                f'\tRegularized Dirichlet iter: {it} objective: {new_li:.4g} gradient norm: {g_norm:.4g}')
        if new_li > old_li:
            logging.info(
                f' Objective not reducing iter: {it} old: {old_li:.4g} new: {new_li:.4g}')
            with open('nrd.dbg.pkl', 'wb') as f:
                pickle.dump((rho, r), f)
            raise Exception('Stopping in admm.newton_regularized_dirichlet.')

        if old_li - new_li < tol:
            break
    if verbosity >= 2:
        logging.info(f'\tRegularized Dirichlet iter: {it:} objective: {new_li:.4g}'
                     f' gradient norm: {g_norm:.4g} sc: {sc:.4g}')
    if it == max_iter - 1:
        logging.warn(' Regularized Dirichlet did not converge.')
        logging.info(f'new_li:{new_li} old_li:{old_li}')
        with open('nrd.dbg.pkl', 'wb') as f:
            pickle.dump((rho, r), f)
        raise Exception('Stopping in admm.newton_regularized_dirichlet.')

    return taus


def update_e(taus, v, rho):
    return taus + 1 / rho * v


def update_xis(es, rho, D, s, max_iter=100, verbosity=0, mu=2, primal_tol=1e-3):
    n, k = es.shape
    l = D.shape[0]
    xis = []
    for i in range(k):
        e = es[:, [i]]
        gamma, _ = primal_dual(e, rho, D, s, max_iter=max_iter, mu=mu,
                               verbosity=verbosity, primal_tol=primal_tol)
        xi, _ = split_gamma(gamma, n, 1, l)
        xis.append(xi)
    return np.concatenate(xis, axis=1)


def update_r(xis, v, cs, rho):
    return xis - 1 / rho * v + 1 / rho * cs


def update_tau(r, rho, verbosity=0, max_iter=20, ls_iter=10):
    new_taus = newton_regularized_dirichlet(
        rho, r, max_iter=max_iter, ls_iter=ls_iter, verbosity=verbosity)
    assert np.all(new_taus > 0)
    return np.reshape(new_taus, r.shape)


def update_v(v, taus, xis, rho):
    return v + rho * (taus - xis)


def primal_objective(taus, cs, s, D):
    n, k = np.shape(cs)
    objective = np.sum(gammaln(taus))
    objective -= np.sum(gammaln(np.sum(taus, axis=1)))
    objective -= np.sum(np.multiply(taus, cs))
    chis = np.abs(D @ taus)
    objective += np.sum(scipy.sparse.diags(s, 0).dot(chis))
    return objective


def admm(cs, D, s, rho, verbosity=0, max_iter=15,
         max_dirichlet_iter=20, max_dirichlet_ls_iter=10,
         max_primal_dual_iter=400,
         mu=2, primal_tol=1e-3, threshold=None):
    """Performs an ADMM update to optimize per-cell topic prior Xi given LDA parameters.

    Reference: Modeling Multiplexed Images with Spatial-LDA Reveals Novel Tissue Microenvironments.

    This performs the update for Xi (refer to eqn. 5 in the appendix) which is alternated with the modified
    LDA fit (which optimizes phi, gamma and lambda).

    Args:
        cs: C_ik = Digamma(gamma_ik) - Digamma(sum_k gamma_ik) where gamma is the unnormalized topic preference of
            cell i for topic k.
        D: Difference matrix that encodes pairs of cells that should be regularized to be similar
           (see: featurization.make_merged_difference_matrix). This should have shape (num_edges x num_cells).
        s: Difference penalty for each edge / pair of cells that should be regularized to be similar.
           This should have shape (num_edges). In the paper this is denoted (1 / d_ij). The larger s is, the more
           strongly adjacent cells are forced to agree on their topic priors.
        rho: ADMM parameter controlling the strength of the consensus term. Higher value of rho force the independent
             Xis to converge more quickly to a common consensus.
        verbosity: Whether to print debugging output.
        max_iter: Maximum number of ADMM iterations to run.
        max_primal_dual_iter: Maximum number of primal-dual iterations to run.
        max_dirichlet_iter: Maximum number of newton steps to take in computing updates for tau (see 5.2.8 in the
                            appendix).
        max_dirichlet_ls_iter: Maximum number of line-search steps to take in computing updates for tau
                               (see 5.2.8 in the appendix).
        primal_tol: tolerance level for primal-dual updates.
        threshold: Cutoff for the percent change in the objective function.  Typical value is
            0.01.  If None, then all iterations in max_iter are executed.
    Returns:
        Xi (see section 2.4 in the reference).
    """
    if threshold is not None:
        assert 0 < threshold < 1
    taus = np.ones(cs.shape)
    xis = np.ones(cs.shape)
    v = np.zeros(cs.shape)
    start = time.time()
    for i in range(max_iter):
        es = update_e(taus, v, rho)
        start_xis = time.time()
        xis_old, taus_old = xis, taus
        xis = update_xis(es, rho, D, s, max_iter=max_primal_dual_iter,
                         verbosity=verbosity, mu=mu, primal_tol=primal_tol)
        if verbosity >= 1:
            duration = time.time() - start_xis
            logging.info(f'\tADMM Primal-Dual Fusion took:{duration:.2f} seconds')
        r = update_r(xis, v, cs, rho)
        start_tau = time.time()
        taus = update_tau(
            r,
            rho,
            max_iter=max_dirichlet_iter,
            ls_iter=max_dirichlet_ls_iter,
            verbosity=verbosity)
        if verbosity >= 1:
            duration = time.time() - start_tau
            logging.info(
                f'\tADMM Newton Regularized Dirichlet took:{duration:.2f} seconds')
        v = update_v(v, taus, xis, rho)
        primal_residual = np.linalg.norm(taus - xis)
        dual_residual = rho * (np.linalg.norm(xis_old - xis) +
                               np.linalg.norm(taus_old - taus))
        residual_ratio = primal_residual / dual_residual

        if residual_ratio > ADMM_RESIDUAL_RATIO_BOUND:
            rho *= ADMM_RHO_SCALE
        elif residual_ratio < 1 / ADMM_RESIDUAL_RATIO_BOUND:
            rho /= ADMM_RHO_SCALE

        objective_old = primal_objective(taus_old, cs, s, D)
        objective = primal_objective(taus, cs, s, D)
        pct_change = abs(objective_old - objective) / objective_old

        if verbosity >= 1:
            norm_v = np.linalg.norm(v)
            duration = time.time() - start
            logging.info(f'\nADDM it:{i} primal res.:{primal_residual:.5g}'
                         f' dual res.:{dual_residual:.5g}.'
                         f' norm of v:{norm_v:.5g}'
                         f' objective: {objective:.5g}'
                         f' old objective: {objective_old:.5g}'
                         f' percent change: {pct_change:.5g}'
                         f' rho: {rho:.5f}'
                         f' Time since start:{duration:.2f} seconds\n')

        if threshold is not None:
            if pct_change < threshold:
                break

    return xis
