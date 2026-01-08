import numpy as np
import random
from typing import List


class KMeansCustom:
    def __init__(self, cluster_cnt: int, max_iter: int = 300):
        self.cluster_cnt = cluster_cnt
        self.max_iter = max_iter

    def initialize_clusters(self, arr: np.ndarray) -> np.ndarray:
        n_features = arr.shape[1]
        max_values = arr.max(axis=0)
        min_values = arr.min(axis=0)
        clusters = np.array([
            [random.uniform(min_values[d], max_values[d]) for d in range(n_features)]
            for _ in range(self.cluster_cnt)
        ])
        return clusters

    def assign_clusters(self, arr: np.ndarray, clusters: np.ndarray):
        cluster_values = [[] for _ in range(self.cluster_cnt)]
        labels = []
        for i in range(arr.shape[0]):
            distances = [np.linalg.norm(arr[i] - clusters[j]) for j in range(self.cluster_cnt)]
            nearest = np.argmin(distances)
            cluster_values[nearest].append(arr[i])
            labels.append(nearest)
        return cluster_values, labels

    def update_clusters(self, clusters: np.ndarray, cluster_values: list) -> np.ndarray:
        new_clusters = []
        for i in range(self.cluster_cnt):
            if cluster_values[i]:
                new_clusters.append(np.mean(cluster_values[i], axis=0))
            else:
                new_clusters.append(clusters[i])
        return np.array(new_clusters)

    def fit(self, X: np.ndarray):
        arr = np.array(X)
        clusters = self.initialize_clusters(arr)

        for _ in range(self.max_iter):
            cluster_values, labels = self.assign_clusters(arr, clusters)
            new_clusters = self.update_clusters(clusters, cluster_values)
            if np.allclose(clusters, new_clusters):
                break
            clusters = new_clusters

        self.cluster_centers_ = clusters
        self.labels_ = np.array(labels)

        self.inertia_ = 0
        for i in range(arr.shape[0]):
            center = self.cluster_centers_[self.labels_[i]]
            self.inertia_ += np.linalg.norm(arr[i] - center) ** 2

    def predict(self, X: np.ndarray) -> List[int]:
        X = np.array(X)
        preds = []
        for x in X:
            distances = [np.linalg.norm(x - c) for c in self.cluster_centers_]
            preds.append(np.argmin(distances))
        return preds
