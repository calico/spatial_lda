from collections import namedtuple
import itertools
import os
import shutil
import tempfile
import unittest
import logging
import time
import numpy as np
import scipy.optimize
from scipy.special import gammaln

from spatial_lda.admm import ALPHA
from spatial_lda.admm import admm
from spatial_lda.admm import assemble_block_diag
from spatial_lda.admm import build_linear_system
from spatial_lda.admm import compute_r
from spatial_lda.admm import line_search
from spatial_lda.admm import make_A
from spatial_lda.admm import make_C
from spatial_lda.admm import f0
from spatial_lda.admm import gap
from spatial_lda.admm import get_update_step
from spatial_lda.admm import gradient_f0
from spatial_lda.admm import gradient_li
from spatial_lda.admm import hessian_f0
from spatial_lda.admm import hessian_li
from spatial_lda.admm import newton_regularized_dirichlet
from spatial_lda.admm import li
from spatial_lda.admm import split_gamma
from spatial_lda.admm import primal_dual
from spatial_lda.admm import primal_objective
from spatial_lda.primal_dual import primal_dual as reference_solver


def make_chain_matrix(n):
    D = np.zeros((n - 1, n))
    for i in range(n - 1):
        D[i, i] = 1
        D[i, i + 1] = -1
    return D


def make_regularized_dirichlet_example(n, k, seed):
    np.random.seed(seed)
    true_r = np.arange(1.0, k + 1.0)
    r = np.vstack([true_r] * n)
    taus = r + 0.01 * np.random.normal(size=(n, k))
    return taus, r


def make_fusion_example(n, k, seed):
    np.random.seed(seed)
    true_xi = np.arange(1.0, k + 1.0)
    xis = np.vstack([true_xi] * n)
    true_xis = xis.copy()
    xis += np.random.uniform(size=(n, k)) - 0.5
    D = make_chain_matrix(n)
    e = true_xis + 0.01 * np.random.normal(size=(n, k))
    chis = np.abs(np.dot(D, xis) + 0.1)
    true_chis = np.abs(np.dot(D, true_xis))
    return true_xis, true_chis, xis, chis, e, D


def make_spatial_lda_example(n, k, seed):
    np.random.seed(seed)
    true_xi = np.arange(1.0, k + 1.0)
    xis = np.vstack([true_xi] * n)
    true_xis = xis.copy()
    xis += 0.1 * (np.random.uniform(size=(n, k)) - 0.5)
    D = make_chain_matrix(n)
    cs = np.log(np.random.dirichlet(true_xi, n))
    cchis = np.abs(np.dot(D, xis) + 0.1)
    true_cchis = np.abs(np.dot(D, true_xis))
    return true_xis, true_cchis, xis, cchis, cs, D


def make_gamma(xi, chi):
    xi_flat = np.reshape(xi, [np.prod(xi.shape), 1])
    chi_flat = np.reshape(chi, [np.prod(chi.shape), 1])
    return np.vstack((xi_flat, chi_flat))


def finite_differences_gradient(f0, x0, delta=1e-8):
    assert x0.shape[0] == np.prod(x0.shape)
    fd = np.zeros(x0.shape)
    for i in range(x0.shape[0]):
        x0[i] -= delta / 2
        f1 = f0(x0)
        x0[i] += delta
        f2 = f0(x0)
        x0[i] -= delta / 2
        fd[i] = (f2 - f1) / delta
    return fd


def finite_differences_hessian(gradient_f0, x0, delta=1e-8):
    assert x0.shape[0] == np.prod(x0.shape)
    d = x0.shape[0]
    Hfd = np.zeros((d, d))
    for i in range(d):
        Hfd[i, :] = finite_differences_gradient(
            lambda x: gradient_f0(x)[i], x0, delta=delta).squeeze()
    return Hfd


class TestADMM(unittest.TestCase):
    def test_split_gamma(self):
        n, k = 4, 5
        _, _, xis, chis, _, D = make_fusion_example(n, k, seed=1)
        l = D.shape[0]
        gamma = make_gamma(xis, chis)
        xis2, chis2 = split_gamma(gamma, n=n, k=k, l=l)
        np.testing.assert_almost_equal(xis, xis2)
        np.testing.assert_almost_equal(chis, chis2)

    def test_make_A(self):
        n, k = 4, 2
        _, _, xis, _, _, D = make_fusion_example(n, k, seed=1)
        l = D.shape[0]
        A = make_A(D, k)
        np.testing.assert_almost_equal(
            np.ravel(D @ xis), A @ np.ravel(xis))

    def test_make_C(self):
        pass

    def test_f0(self):
        n, k = 4, 5
        true_xis, true_chis, xis, chis, e, D = make_fusion_example(
            n, k, seed=1)
        gamma = make_gamma(xis, chis)
        true_gamma = make_gamma(true_xis, true_chis)
        l = D.shape[0]
        rho = 0.1
        o1 = f0(gamma, e, rho, np.zeros(l))
        o2 = f0(true_gamma, e, rho, np.zeros(l))
        self.assertLess(o2, o1)

    def test_gradient_f0(self):
        n, k = 3, 4
        true_xis, true_chis, xis, chis, e, D = make_fusion_example(
            n, k, seed=1)
        gamma = make_gamma(xis, chis)
        l = D.shape[0]
        s = np.ones(l)
        rho = 0.1
        g = gradient_f0(gamma, e, rho, s)
        fd = finite_differences_gradient(lambda x0: f0(x0, e, rho, s), gamma)
        delta = 1e-8
        np.testing.assert_almost_equal(fd, g, decimal=5)

    def test_hessian_f0(self):
        n, k = 2, 3
        true_xis, true_chis, xis, chis, e, D = make_fusion_example(
            n, k, seed=1)
        gamma = make_gamma(xis, chis)
        l = D.shape[0]
        s = np.ones(l)
        rho = 0.1
        H = hessian_f0(gamma, e, rho, s)
        H = H.toarray()
        Hfd = finite_differences_hessian(
            lambda x0: gradient_f0(
                x0, e, rho, s), gamma)
        np.testing.assert_almost_equal(Hfd, H, decimal=5)
        v, _ = np.linalg.eig(H)
        np.testing.assert_array_less(-v, 1e-8)

    def test_build_linear_system(self):
        n, k = 4, 3
        true_xis, true_chis, xis, chis, e, D = make_fusion_example(
            n, k, seed=1)
        gamma = make_gamma(xis, chis)
        l = D.shape[0]
        s = np.ones(l)
        u = np.ones(2 * l * k)
        t, rho = 0.1, 0.1
        C = make_C(D, k)
        gamma = make_gamma(xis, chis)
        M, r = build_linear_system(gamma, u, C, e, rho, s, t)
        self.assertEqual(M.shape[0], n * k + l * k + 2 * l * k)
        self.assertEqual(M.shape[0], r.shape[0])
        self.assertEqual(M.shape[0], M.shape[1])
        delta = scipy.sparse.linalg.spsolve(M, r)
        np.testing.assert_array_less(np.abs(delta), 10)

    def test_gap(self):
        n, k = 4, 3
        true_xis, true_chis, xis, chis, e, D = make_fusion_example(
            n, k, seed=1)
        gamma = make_gamma(xis, chis)
        l = D.shape[0]
        u = np.ones(2 * l * k)
        C = make_C(D, k)
        self.assertLess(gap(gamma, C, u), 10)

    def test_line_search(self):
        n, k = 4, 3
        true_xis, true_chis, xis, chis, e, D = make_fusion_example(
            n, k, seed=1)
        gamma = np.squeeze(make_gamma(xis, chis))
        l = D.shape[0]
        s = np.ones(l)
        u = np.ones(2 * l * k)
        C = make_C(D, k)
        t = 2 * (l * k) / gap(gamma, C, u)
        rho = 0.1
        new_gamma, new_u, step, _ = line_search(gamma, u, C, e, rho, s, t, l)
        r1 = compute_r(gamma, u, C, e, rho, s, t)
        r2 = compute_r(new_gamma, new_u, C, e, rho, s, t)
        self.assertLess(
            np.linalg.norm(r2),
            (1 - ALPHA * step) * np.linalg.norm(r1))

    def test_primal_dual(self):
        n, k = 100, 3
        true_xis, true_chis, _, _, e, D = make_fusion_example(n, k, seed=1)
        l = D.shape[0]
        s = 100.0 * np.ones(l)
        rho = 1e-6
        gamma, u = primal_dual(e, rho, D, s)
        xis, chis = split_gamma(gamma, n, k, l)
        np.testing.assert_array_almost_equal(xis, true_xis, decimal=2)
        np.testing.assert_array_less(np.abs(chis), 1e-6)

    def test_li(self):
        n, k = 4, 3
        true_taus, r = make_regularized_dirichlet_example(n, k, seed=1)
        taus = true_taus + 0.1 * np.random.normal(size=(n, k))
        rho = 1.0
        ll1 = li(taus, r, rho)
        ll2 = li(true_taus, r, rho)
        self.assertLess(ll2, ll1)

    def test_gradient_li(self):
        n, k = 4, 3
        true_taus, r = make_regularized_dirichlet_example(n, k, seed=1)
        taus = true_taus + 0.1 * np.random.normal(size=(n, k))
        rho = 1.0
        r = r.reshape((-1, 1))
        taus = taus.reshape((-1, 1))
        g = gradient_li(taus, r, rho)
        fd = finite_differences_gradient(lambda x0: li(x0, r, rho), taus)
        np.testing.assert_almost_equal(fd, g, decimal=5)

    def test_assemble_block_diag(self):
        ones = np.ones((2, 2))
        mats = (ones, 2 * ones)
        M = [[1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [0., 0., 2., 2.],
             [0., 0., 2., 2.]]
        np.testing.assert_almost_equal(M, assemble_block_diag(mats).todense())

    def test_hessian_li(self):
        n, k = 4, 3
        true_taus, r = make_regularized_dirichlet_example(n, k, seed=1)
        taus = true_taus + 0.1 * np.random.normal(size=(n, k))
        rho = 1.0
        r = r.reshape((-1, 1))
        taus = taus.reshape((-1, 1))
        H = hessian_li(taus, r, rho)
        Hfd = finite_differences_hessian(
            lambda x0: gradient_li(x0, r, rho), taus)
        np.testing.assert_almost_equal(Hfd, H.todense(), decimal=5)

    def test_get_update_step(self):
        n, k = 4, 3
        true_taus, r = make_regularized_dirichlet_example(n, k, seed=1)
        taus = true_taus + 0.1 * np.random.normal(size=(n, k))
        rho = 1.0
        r = r.reshape((-1, 1))
        taus = taus.reshape((-1, 1))
        H = hessian_li(taus, r, rho).todense()
        g = gradient_li(taus, r, rho)
        newton_direction = np.linalg.pinv(H) @ g
        step, _, _ = get_update_step(taus, r, rho)
        np.testing.assert_almost_equal(step, newton_direction)

    def test_newton_regularized_dirichlet(self):
        n, k = 10, 3
        true_taus, r = make_regularized_dirichlet_example(n, k, seed=1)
        taus = true_taus + 0.1 * np.random.normal(size=(n, k))
        rho = 1.0
        opt_taus = newton_regularized_dirichlet(rho, r)
        ll1 = li(taus, r, rho)
        ll2 = li(opt_taus, r, rho)
        g = gradient_li(opt_taus, r, rho)
        self.assertLess(ll2, ll1)
        self.assertLess(np.linalg.norm(g), 1e-4)

    def test_admm(self):
        n, k = 100, 3
        true_xis, true_cchis, xis, cchis, cs, D = make_spatial_lda_example(
            n, k, seed=1)
        l = D.shape[0]
        s = 1000 * np.ones(l)
        rho = 0.1
        xis = admm(cs=cs, D=D, s=s, rho=rho, verbosity=0, max_iter=40,
                   max_dirichlet_iter=20)
        normed_xis = xis / np.sum(xis, axis=1, keepdims=True)
        normed_true_xis = true_xis / np.sum(true_xis, axis=1, keepdims=True)
        admm_objective = primal_objective(xis, cs, s, D)
        true_params_objective = primal_objective(true_xis, cs, s, D)
        np.testing.assert_almost_equal(normed_xis, normed_true_xis, decimal=2)
        np.testing.assert_array_less(admm_objective, true_params_objective)

        # test adaptive speed improvements
        # baseline: max_iters=50, no threshold, default tolerance
        baseline_start = time.time()
        base_xis = admm(cs=cs, D=D, s=s, rho=rho, verbosity=0, max_iter=50,
                   max_dirichlet_iter=20)
        baseline_time = time.time() - baseline_start
        baseline_objective = primal_objective(base_xis, cs, s, D)

        # percent change in objective threshold
        thresh_start = time.time()
        thresh_xis = admm(cs=cs, D=D, s=s, rho=rho, verbosity=1, max_iter=50,
                       max_dirichlet_iter=20, threshold=0.05)
        thresh_time = time.time() - thresh_start
        thresh_objective = primal_objective(thresh_xis, cs, s, D)
        np.testing.assert_array_less(thresh_objective, true_params_objective)
        np.testing.assert_array_less(thresh_time, baseline_time)

        # test primal-dual tolerance level
        tolerance_start = time.time()
        tol_xis = admm(cs=cs, D=D, s=s, rho=rho, verbosity=1, max_iter=50,
                       max_dirichlet_iter=20, primal_tol=0.01)
        tolerance_time = time.time() - tolerance_start
        tolerance_objective = primal_objective(tol_xis, cs, s, D)
        np.testing.assert_almost_equal(baseline_objective, tolerance_objective, decimal=2)
        np.testing.assert_array_less(tolerance_time, baseline_time)

if __name__ == '__main__':
    unittest.main()

