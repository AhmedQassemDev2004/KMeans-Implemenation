from sklearn.datasets import make_blobs
import numpy as np
from KMeans import KMeans

np.random.seed(42)

X, y = make_blobs(
    centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
)
print(X.shape)

clusters = len(np.unique(y))
print(clusters)

k = KMeans(K=clusters, max_iterations=150, plot_steps=False)
y_pred = k.predict(X)

k.plot()
