# -*- coding: utf-8 -*-
import numpy as np
from numba import njit
from scipy import stats
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


def get_gaussian_fun(mu, cov):
    """
    Get a function that calculate n dimensional gaussian function from
        the mean and the covariance matrix

    Args:
        mu (np.ndarray): the mean of the gaussian distribution, shape (n, 1)
        cov (np.ndarray): the covariance matrix of the gaussian distribution
            the shape should be (n, n)

    Return:
        callable: the gaussian function f, where f(X) gives the probabilities
            of sample X
    """
    c = np.linalg.det(cov)
    n = len(mu)
    A = 1 / ((2 * np.pi)**(n / 2) * c**0.5)
    icov = np.linalg.inv(cov)
    return lambda X: A * np.exp(
        -0.5 * np.sum(
            np.dot((X - mu).T,  icov) * (X - mu).T,
            axis=1
        )
    )


@njit
def get_spatial_descriptors(positions, image, radius):
    """
    Get the spatial descriptors for the voxels around
        each particles inside a sphere.

    - dx, dy, dz: the shift along different axes from the particle centr
    - dx2, dy2, dz2: the squared shift along different axes from the particle centre
    - dr2xy: the squared radius from the centre in the xy-plane
    - dr2: the squared radius from the centre

    Args:
        positions (np.ndarray): the location of particles, shape (n, 3)
        image (np.ndarray): the 3D volumetric image being tracked
        radius (int): the radius inside witch the voxels will be considered

    Return:
        np.ndarray: spatial descriptors, shape (9, n),
            1, dx, dy, dz, dx2, dy2, dz2, dr2xy, dr2
    """
    r2_max = radius * radius
    x_max, y_max, z_max = image.shape
    descriptors = np.empty((
        9, len(positions), (2 * radius + 1) ** 3
    ))
    for idx, xyz in enumerate(positions):
        x, y, z = xyz
        count = 0
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                for k in range(-radius, radius+1):
                    r2 = i**2 + j**2 + k**2
                    should_count = r2 <= r2_max
                    should_count *= x + i >= 0
                    should_count *= y + j >= 0
                    should_count *= z + k >= 0
                    should_count *= x + i < x_max
                    should_count *= y + j < y_max
                    should_count *= z + k < z_max
                    if should_count:
                        i2, j2, k2 = i*i, j*j, k*k
                        descriptors[0, idx, count] = 1
                        descriptors[1, idx, count] = i
                        descriptors[2, idx, count] = j
                        descriptors[3, idx, count] = k
                        descriptors[4, idx, count] = i2
                        descriptors[5, idx, count] = j2
                        descriptors[6, idx, count] = k2
                        descriptors[7, idx, count] = i2 + j2
                        descriptors[8, idx, count] = i2 + j2 + k2
                        count += 1
    return descriptors


@njit
def get_intensities(positions, image, radius):
    """
    Get the voxel intensities around a particle inside a sphere

    Args:
        positions (np.ndarray): the locations of particles, shape (n, 3)
        image (np.ndarray): the 3D volumetric image to be tracked
        radius (int): the radius of the sphere withen which the voxel
            intensities will be collected

    Return:
        np.ndarray: the voxel intensities for each particle, shape (n, diameter^3)
            there are **NAN** in the result for voxels in the diameter^3
            cubic but outside the sphere
    """
    r2_max = radius * radius
    x_max, y_max, z_max = image.shape
    intensities = np.empty((len(positions), (2 * radius + 1) ** 3))
    intensities[:] = np.nan
    for idx, xyz in enumerate(positions):
        x, y, z = xyz
        x, y, z = int(x), int(y), int(z)
        count = 0
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                for k in range(-radius, radius + 1):
                    r2 = i**2 + j**2 + k**2
                    should_count = r2 <= r2_max
                    should_count *= x + i >= 0
                    should_count *= y + j >= 0
                    should_count *= z + k >= 0
                    should_count *= x + i < x_max
                    should_count *= y + j < y_max
                    should_count *= z + k < z_max
                    if should_count:
                        intensities[idx, count] = image[x+i, y+j, z+k]
                        count += 1
    return intensities


def get_nn_distances(positions):
    """
    Get the nearest neighbour distances of different particles

    Args:
        positions (np.ndarray): locations of particles, shape (n, 3)

    Return:
        np.ndarray: the nearest neighbour distances, shape (n, )
    """
    dist_mat = squareform(pdist(positions))
    np.fill_diagonal(dist_mat, np.nan)
    return np.nanmin(dist_mat, axis=0)


def get_gaussian_par(features):
    """
    Get the mean and variance for gaussian fit

    Args:
        features (np.ndarray): shape (n_features, n_samples)
    """
    mu = np.mean(features, axis=1)[:, np.newaxis]
    cov = np.dot((features - mu), (features - mu).T)
    return mu, cov


def feature_plot_2d(X):
    """
    Plot the 2D gaussian fit of two features

    Args:
        X (np.ndarray): two features shape (2, n)
    """
    x = np.linspace(X[0].min() - X[0].std(), X[0].max() + X[0].std(), 100)
    y = np.linspace(X[1].min() - X[1].std(), X[1].max() + X[1].std(), 100)
    x, y = np.meshgrid(x, y)
    mu, cov = get_gaussian_par(X)
    func = get_gaussian_fun(mu, cov)
    z = func(np.array((x.ravel(), y.ravel())))
    z = z.reshape(100, 100)
    fig, ax = plt.subplots()
    ax.scatter(*X, marker='+', color='k', alpha=0.5)
    cs = ax.contour(x, y, z)
    ax.clabel(cs, inline=1, fontsize=12, fmt='%1.6f')
    plt.xlabel("Feature 1", fontsize=16)
    plt.ylabel("Feature 2", fontsize=16)
    plt.show()


def reduce_dimension(features, percentage=0.95):
    """
    Use the principle component analysis (PCA) to reduce the dimension
        of the features. The final dimension is determined by retaining
        a fixed percentage of variance.

    Args:
        features (np.ndarray): the shape is (n_feature, n_sample)
        percentage (float): the percentage of variance to be retained, (0 ~ 1)

    Return:
        np.ndarray: the features in a reduced dimensional space,
            the shape is (n_feature_reduced, n_sample)
    """
    cov = np.dot(features, features.T) / features.shape[1]
    u, s, vh = np.linalg.svd(cov)
    variance_retained = 1 - s / s.sum()
    to_retain = (variance_retained <= percentage).sum() + 1
    u_reduced = u[:, :to_retain]
    return np.dot(u_reduced.T, features)


class ParticleFeatures:
    __stat_methods = [
        lambda x : np.nanmean(x, axis=1),
        lambda x : np.nanvar(x, axis=1),
        lambda x : stats.skew(x, axis=1, nan_policy='omit'),
        lambda x : stats.kurtosis(x, axis=1, nan_policy='omit'),
    ]

    def __init__(self, positions, image, radius):
        self.positions = positions
        self.image = image
        self.radius = radius
        self.__n = len(positions)
        self.__spatial = get_spatial_descriptors(positions, image, radius)
        self.__intensities = get_intensities(positions, image, radius)
        self.nnd = get_nn_distances(positions)
        self.features = self.__get_features()
        self.m0 = np.nansum(self.__intensities, axis=1)
        self.m2 = np.nansum(self.__spatial[8] * self.__intensities, axis=1) / self.m0

    def __get_features(self):
        """
        Get a feature matrix whose rol/column have following meanings

        ..code-block::

           dr2 * intensity  ───────────────────┐
         dr2xy * intensity  ─────────────────┐ │
           dz2 * intensity  ───────────────┐ │ │
           dy2 * intensity  ─────────────┐ │ │ │
           dx2 * intensity  ───────────┐ │ │ │ │
            dz * intensity  ─────────┐ │ │ │ │ │
            dy * intensity  ───────┐ │ │ │ │ │ │
            dx * intensity  ─────┐ │ │ │ │ │ │ │
                 intensity  ───┐ │ │ │ │ │ │ │ │
                            ┌─┬▼─▼─▼─▼─▼─▼─▼─▼─▼─┐
                            │ │0 1 2 3 4 5 6 7 8 │
                            ├─┼──────────────────┤
                      mean  │0│f00    ....   f08 │
                  variance  │1│        .         │
                      skew  │2│         .        │
                  kurtosis  │3│f30    ....   f38 │
                            └─┴──────────────────┘

        If all values in the distribution were positive,
            take the logarithm as feature values
            (the log distributions are more gaussian)
        """
        features = np.empty((4, 9, self.__n))
        for row, method in enumerate(self.__stat_methods):
            for col, spatial in enumerate(self.__spatial):
                f = method(spatial * self.__intensities)
                if f.min() > 0:
                    f = np.log(f)
                features[row, col] = (f - f.mean()) / (f.max() - f.min())
        return features

    def get_probability(self):
        features = np.reshape(self.features, (36, self.__n))
        mu, cov = get_gaussian_par(features)
        return get_gaussian_fun(mu, cov)(features)

    def get_probability_reduced(self, percentage=95):
        features = np.reshape(self.features, (36, self.__n))
        features_reduced = reduce_dimension(features, percentage)
        print(features_reduced.shape)
        mu, cov = get_gaussian_par(features_reduced)
        return get_gaussian_fun(mu, cov)(features_reduced)
