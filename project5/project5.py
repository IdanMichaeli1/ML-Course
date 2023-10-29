import numpy as np


def get_random_centroids(X, k):
    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    np.random.seed(42)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    return np.asarray(centroids).astype(np.float)


def lp_distance(X, centroids, p=2):
    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    k = len(centroids)
    num_pixels = X.shape[0]
    diff = X.reshape(num_pixels, 1, 3) - centroids.reshape(1, k, 3)
    return np.linalg.norm(diff, ord=p, axis=2).T


def kmeans(X, k, p, max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    centroids = get_random_centroids(X, k)
    centroids, classes = kmeans_loop(X, k, p, centroids, max_iter=100)

    return centroids, classes


def kmeans_pp(X, k, p, max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    num_pixels = X.shape[0]
    centroids = np.empty((k, 3))
    centroids[0] = X[np.random.choice(num_pixels)]

    for i in range(1, k):
        distances = lp_distance(X, centroids[:i], p)
        min_distances = np.min(distances, axis=0)
        probabilities = min_distances ** 2 / np.sum(min_distances ** 2)
        new_centroid_index = np.random.choice(num_pixels, p=probabilities)
        centroids[i] = X[new_centroid_index]

    centroids, classes = kmeans_loop(X, k, p, centroids, max_iter=100)

    return centroids, classes


def kmeans_loop(X, k, p, centroids, max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = np.zeros(X.shape[0])
    prev_centroids = centroids.copy()

    for _ in range(max_iter):
        distances = lp_distance(X, centroids, p)
        classes = np.argmin(distances, axis=0)

        for i in range(k):
            cluster_points = X[classes == i]
            centroids[i] = np.mean(cluster_points, axis=0)

        if np.array_equal(centroids, prev_centroids):
            break

        prev_centroids = centroids.copy()

    return centroids, classes
