import numpy as np


def weighted_std(xs, weights):
    weighted_mean = np.average(xs, weights=weights)
    weighted_var = (np.sum(weights * (xs - weighted_mean) ** 2)
                    / np.sum(weights))
    return weighted_var ** 0.5


def test_weighted_std():
    xs = np.random.rand(100)
    wstd = weighted_std(xs, weights=[1] * 100)
    np.testing.assert_allclose(wstd, np.std(xs))
    wstd = weighted_std(xs, weights=[0.1] * 100)
    np.testing.assert_allclose(wstd, np.std(xs))
    assert wstd == np.std(xs)
    wstd = weighted_std(xs, weights=[0.1] * 50 + [0] * 50)
    np.testing.assert_allclose(wstd, np.std(xs[:50]))


if __name__ == '__main__':
    test_weighted_std()
