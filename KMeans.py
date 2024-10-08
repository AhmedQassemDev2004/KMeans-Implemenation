import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, K=5, max_iterations=100, plot_steps=False) -> None:
        self.K = K
        self.max_iterations = max_iterations
        self.plit_steps = plot_steps

        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        random_samples_idxs = np.random.choice(
            self.n_samples, self.K, replace=False)

        self.centroids = [X[idx] for idx in random_samples_idxs]

        # optimization
        for _ in range(self.max_iterations):
            # assign points to the closest centroid
            self.clusters = self._create_clusters(self.centroids)

            if self.plit_steps:
                self.plot()

            # Calculate new centroids from the clusters
            old_centroids = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(old_centroids, self.centroids):
                break

        return self._get_cluster_labels(self.clusters)

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [self._distance(sample, point) for point in centroids]
        return np.argmin(distances)

    def _distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, old_centroids, centroids):
        for i in range(self.K):
            if self._distance(old_centroids[i], centroids[i]) > 0:
                return False

        return True

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()
