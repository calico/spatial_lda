import logging

import numpy as np
from scipy.special import gammaln, digamma, polygamma
import scipy.sparse
import scipy.sparse.linalg

# Line-search parameters
ALPHA = 0.1
BETA = 0.5
MAXLSITER = 50

# Primal-dual iteration parameters
MU = 1e-3
MAXITER = 500
TOL = 1e-2


def make_gamma(xi, chi):
    xi_flat = np.reshape(xi, [np.prod(xi.shape), 1])
    chi_flat = np.reshape(chi, [np.prod(chi.shape), 1])
    return np.vstack((xi_flat, chi_flat))


def split_gamma(gamma, n, k, l):
    xi = np.reshape(gamma[0:n * k], [n, k])
    chi = np.reshape(gamma[n * k: n * k + l * k], [l, k])
    return xi, chi


def make_A(D, k):
    return scipy.sparse.kron(D, np.eye(k))


def make_C(D, k, l):
    """Given differencing matrix on samples D construct C."""
    A = make_A(D, k)
    I = scipy.sparse.eye(l * k)
    As = scipy.sparse.vstack((A, -A))
    Is = scipy.sparse.vstack((-I, -I))
    return scipy.sparse.hstack((As, Is)).tocsr()


def f0(gamma, c, s):
    """Compute f0 for primal vars xi and chi, counts c, and edge weights s"""
    # n documents, k topics, l ties between documents
    n, k = np.shape(c)
    l = np.shape(s)[0]
    xi, chi = split_gamma(gamma, n, k, l)
    objective = 1 / n * np.sum(gammaln(xi))
    objective -= 1 / n * np.sum(gammaln(np.sum(xi, axis=1)))
    objective -= 1 / n * np.sum(np.multiply(xi, c))
    objective += np.sum(scipy.sparse.diags(s, 0).dot(chi))
    return objective


def gradient_f0(gamma, c, s):
    """Compute gradient of objective given variable gamma, counts c, and edge weights s."""
    n, k = np.shape(c)
    l = np.shape(s)[0]
    xi, chi = split_gamma(gamma, n, k, l)
    gxi = 1 / n * \
        (digamma(xi) - digamma(np.sum(xi, axis=1, keepdims=True)) - c)
    gchi = scipy.sparse.diags(s, 0).dot(np.ones((l, k)))
    gxi = np.reshape(gxi, (n * k, 1))
    gchi = np.reshape(gchi, (l * k, 1))
    return np.vstack((gxi, gchi))


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


def hessian_f0(gamma, n, k, l):
    """Compute Hessian of objective given xi and count of edges"""
    xi, chi = split_gamma(gamma, n, k, l)
    blocks = []
    for i in range(n):
        block = np.diag(polygamma(1, xi[i, :])) - \
            polygamma(1, np.sum(xi[i, :]))
        blocks.append(block)
    # nabla2_xi = 1/n*scipy.sparse.block_diag(blocks)
    nabla2_xi = 1 / n * assemble_block_diag(blocks)
    zeros_nk_lk = scipy.sparse.coo_matrix((n * k, l * k))
    zeros_lk_lk = scipy.sparse.coo_matrix((l * k, l * k))
    H = scipy.sparse.vstack((scipy.sparse.hstack((nabla2_xi, zeros_nk_lk)),
                             scipy.sparse.hstack((zeros_nk_lk.T, zeros_lk_lk))))
    return H


def r_dual(gamma, u, C, cs, s):
    g = gradient_f0(gamma, cs, s)
    r = np.squeeze(g) + np.squeeze((C.T.dot(u)))
    return r


def r_cent(gamma, u, C, t):
    f1 = C.dot(gamma)
    return -np.squeeze(scipy.sparse.diags(u, 0).dot(f1)) - 1. / t


def compute_r(gamma, u, C, cs, s, t):
    r1 = r_dual(gamma, u, C, cs, s)
    r2 = r_cent(gamma, u, C, t)
    r = -np.hstack((r1, r2))
    return r


def build_linear_system(gamma, u, C, cs, s, t):
    n, k = cs.shape
    l = u.shape[0] // (2 * k)
    H = hessian_f0(gamma, n, k, l)
    uC = scipy.sparse.diags(np.squeeze(u), 0).dot(C)
    Cg = scipy.sparse.diags(np.squeeze(C.dot(gamma)))
    M = scipy.sparse.vstack((scipy.sparse.hstack((H, C.T)),
                             scipy.sparse.hstack((-uC, -Cg)))).tocsr()
    r = compute_r(gamma, u, C, cs, s, t)
    return M, r


def split_primal_dual_vars(z, n, k, l):
    gamma = z[:n * k + l * k]
    u = z[n * k + l * k:]
    return np.squeeze(gamma), np.squeeze(u)


def gap(gamma, C, u):
    return -np.sum(C.dot(np.squeeze(gamma)) * np.squeeze(u))


def line_search(gamma, u, C, cs, s, t, n, l, k):
    M, r = build_linear_system(gamma, u, C, cs, s, t)
    delta = scipy.sparse.linalg.spsolve(M, r)
    dgamma, du = split_primal_dual_vars(delta, n, k, l)
    step_max = 1.0
    neg_du = du < 0
    if np.any(neg_du):
        step_max = np.min((step_max, np.min(u[neg_du] / (-du[neg_du]))))

    neg_dgamma = dgamma < 0
    if np.any(neg_dgamma):
        step_max = np.min(
            (step_max, np.min(gamma[neg_dgamma] / (-dgamma[neg_dgamma]))))

    step = step_max * 0.99
    r = compute_r(gamma, u, C, cs, s, t)
    for lsit in range(MAXLSITER):
        new_gamma = gamma + step * dgamma
        new_u = u + step * du
        new_r = compute_r(new_gamma, new_u, C, cs, s, t)
        if (np.any(C.dot(new_gamma) > 0) or
                np.linalg.norm(new_r) > (1 - ALPHA * step) * np.linalg.norm(r)):
            step = step * BETA
        else:
            u = new_u
            gamma = new_gamma
            r = new_r
            break
    if lsit == MAXLSITER - 1:
        logging.warn('Line search failed.')
    return gamma, u, step


def primal_dual(cs, D, s, init_gamma=None,
                init_u=None, verbose=False, tol=TOL):
    l, n = D.shape
    _, k = cs.shape
    assert cs.shape[0] == D.shape[1]
    # gamma = np.hstack((np.ones(n*k), np.ones(l*k)))
    init = (cs / np.sum(cs, axis=1, keepdims=True)).ravel()
    gamma = np.hstack((init, np.ones(l * k)))
    u = 0.01 * np.ones(2 * l * k)
    if init_gamma is not None:
        gamma = init_gamma
    if init_u is not None:
        u = init_u
    C = make_C(D, k, l)
    t = 1.0
    for it in range(MAXITER):
        nu = gap(gamma, C, u)
        t = np.max((2 * MU * l * k * k / nu, t * 1.2))
        gamma, u, step = line_search(gamma, u, C, cs, s, t, n, l, k)
        r = np.linalg.norm(r_dual(gamma, u, C, cs, s))
        xis, chis = split_gamma(gamma, n, k, l)
        if (np.linalg.norm(r) < tol and nu < tol):
            break
        if verbose:
            logging.info('it: {0}, gap: {1}, t: {2}, step: {3}, res: {4}'.format(
                         it, nu, t, step, np.linalg.norm(r)))
    return gamma, u
