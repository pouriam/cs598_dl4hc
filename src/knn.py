from sklearn.neighbors import NearestNeighbors
import numpy as np

image_embeddings = np.load("embeddings.npy")
print(image_embeddings.shape)

if __name__ == "__main__":
    k = 100
    knn = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
    knn.fit(image_embeddings)
    distances, indices = knn.kneighbors(image_embeddings)
    print(distances)
    print(indices)
