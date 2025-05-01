import os
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors

data_dir = "C:/Users/pouria/Documents/Illinois/cs598-dl-for-healthcare/Project/cs598_dl4hc/data/"

# Load embeddings (should be dict {dicom_id: embedding})
embeddings = np.load('../data/embeddings.npy', allow_pickle=True).item()
# test_embeddings = np.load('test_embeddings.npy', allow_pickle=True).item()

# Prepare X_train and X_test arrays
train_df = pd.read_csv(os.path.join(data_dir, 'train.tsv'), sep='\t')
test_df = pd.read_csv(os.path.join(data_dir, 'test.tsv'), sep='\t')

train_ids = set(train_df['dicom_id'])
test_ids = set(test_df['dicom_id'])

# Split the embeddings
train_embeddings = {dicom_id: emb for dicom_id, emb in embeddings.items() if dicom_id in train_ids}
test_embeddings = {dicom_id: emb for dicom_id, emb in embeddings.items() if dicom_id in test_ids}

print("Number of train embeddings:", len(train_embeddings))
print("Number of test embeddings:", len(test_embeddings))

np.save('../data/train_embeddings.npy', train_embeddings)
np.save('../data/test_embeddings.npy', test_embeddings)

train_dicom_ids = list(train_embeddings.keys())
X_train = np.stack([train_embeddings[d] for d in train_dicom_ids])  # shape (N_train, D)

test_dicom_ids = list(test_embeddings.keys())
X_test = np.stack([test_embeddings[d] for d in test_dicom_ids])     # shape (N_test, D)

# Fit KNN on train embeddings
knn = NearestNeighbors(n_neighbors=100, metric='cosine')
knn.fit(X_train)

# Find top 100 nearest neighbors for each test sample
distances, indices = knn.kneighbors(X_test)

# Build the dictionary: {test_dicom_id : [list of 100 nearest train_dicom_ids]}
camera_ready_top100 = {}

for idx, test_dicom_id in enumerate(test_dicom_ids):
    nearest_indices = indices[idx]  # 100 neighbors
    nearest_train_dicom_ids = [train_dicom_ids[i] for i in nearest_indices]
    camera_ready_top100[test_dicom_id] = nearest_train_dicom_ids

# Save it
with open('camera_ready_top100.pkl', 'wb') as f:
    pickle.dump(camera_ready_top100, f)
