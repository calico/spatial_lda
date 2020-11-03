import unittest
import functools
import numpy as np
import pandas as pd
import scipy.signal

from spatial_lda import featurization


def make_grid_df(neighborhood_data, make_all_cells_anchors=False):
    n, m, k = neighborhood_data.shape
    xs, ys = np.meshgrid(range(n), range(n))
    xs = xs.flatten()
    ys = ys.flatten()
    n, m, k = neighborhood_data.shape
    xs, ys = np.meshgrid(range(n), range(n))
    xs = xs.flatten()
    ys = ys.flatten()
    neighborhood_df = pd.DataFrame({'x': xs, 'y': ys, 'is_anchor': make_all_cells_anchors})
    markers = np.reshape(neighborhood_data, [n * m, k])
    neighborhood_df = neighborhood_df.assign(**{'marker-%d' % i : markers[:, i] for i in range(k)})

    xs, ys = np.meshgrid(range(n - 1), range(n - 1))
    xs = xs.flatten()
    ys = ys.flatten()
    anchor_df = pd.DataFrame({'x': xs + 0.5, 'y': ys + 0.5, 'is_anchor': True})
    anchor_df = anchor_df.assign(**{'marker-%d' % i : 1000 for i in range(k)})
    result = pd.concat([anchor_df, neighborhood_df])
    return result


class TestFeaturization(unittest.TestCase):
    def test_featurize_samples(self):
        n_samples = 5
        size = 20
        k = 6
        np.random.seed(42)
        fake_data = np.random.randn(n_samples, size, size, k)
        fake_samples = {'sample-%d' % i : make_grid_df(fake_data[i]) for i in range(n_samples)}
        neighborhood_feature_fn = functools.partial(featurization.neighborhood_to_avg_marker,
                                                    markers=['marker-%d' % i for i in range(k)])
        all_sample_features = featurization.featurize_samples(
                fake_samples, neighborhood_feature_fn, 1, 'is_anchor', 'x', 'y', n_processes=4)
        true_result = []
        for i in range(n_samples):
            convolution = scipy.signal.convolve(fake_data[i], np.ones((2, 2, 1)) / 4, mode='valid')
            true_result.append(convolution.reshape([(size - 1) * (size - 1), k]))
        true_result = np.concatenate(true_result, axis=0)
        np.testing.assert_allclose(true_result, all_sample_features.values)

    def test_serial_featurize_samples(self):
        n_samples = 5
        size = 20
        k = 6
        np.random.seed(42)
        fake_data = np.random.randn(n_samples, size, size, k)
        fake_samples = {'sample-%d' % i : make_grid_df(fake_data[i]) for i in range(n_samples)}
        neighborhood_feature_fn = functools.partial(featurization.neighborhood_to_avg_marker,
                                                    markers=['marker-%d' % i for i in range(k)])
        all_sample_features = featurization.featurize_samples(
                fake_samples, neighborhood_feature_fn, 1, 'is_anchor', 'x', 'y', n_processes=4)
        all_sample_features_serial = featurization.featurize_samples(
                fake_samples, neighborhood_feature_fn, 1, 'is_anchor', 'x', 'y', n_processes=None)

        np.testing.assert_allclose(all_sample_features.values,
                                    all_sample_features_serial.values)
        
        
    def test_all_cells_are_anchors(self):
        n_samples = 5
        size = 20
        k = 6
        np.random.seed(42)
        fake_data = np.random.randn(n_samples, size, size, k)
        fake_samples = {}
        for i in range(n_samples):
            grid_df = make_grid_df(fake_data[i], make_all_cells_anchors=True)
            fake_samples['sample-%d' % i] = grid_df

        neighborhood_feature_fn = functools.partial(featurization.neighborhood_to_avg_marker,
                                                    markers=['marker-%d' % i for i in range(k)])
        
        all_sample_features = featurization.featurize_samples(
                fake_samples, neighborhood_feature_fn, 1, 'is_anchor', 'x', 'y', n_processes=4,
                include_anchors=True)
        
        # all cells are anchors and include_anchors == False, we expect a ValueError
        self.assertRaises(ValueError, lambda: featurization.featurize_samples(
                fake_samples, neighborhood_feature_fn, 1, 'is_anchor', 'x', 'y', n_processes=4,
                include_anchors=False))


if __name__ == '__main__':
    unittest.main()

