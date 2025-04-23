import tensorflow as tf
import pydicom
import numpy as np
import cv2
import os
from tensorflow.keras.applications.densenet import preprocess_input

base_model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, pooling='avg')


def load_dicom_image(dcm_path):
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array.astype(np.float32)

    # Normalize to 0-255
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5) * 255.0
    img = img.astype(np.uint8)

    # Convert to 3-channel RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Resize to 224x224
    img_resized = cv2.resize(img_rgb, (224, 224))

    # Preprocess for DenseNet and return batch shape
    img_preprocessed = preprocess_input(np.expand_dims(img_resized, axis=0))  # shape: (1, 224, 224, 3)
    return img_preprocessed


def create_embedding(dicom_folder):
    embedding_list = []

    for (dirpath, dirnames, filenames) in os.walk(dicom_folder):
        for filename in filenames:
            if filename.lower().endswith('.dcm'):
                try:
                    img_path = os.path.join(dirpath, filename)
                    img = load_dicom_image(img_path)
                    embedding = base_model.predict(img)
                    embedding_list.append(embedding.squeeze())
                except Exception as e:
                    print(e, filename)

    return np.array(embedding_list)


if __name__ == "__main__":
    embeddings = create_embedding("C:/data/physionet/physionet.org/files/mimic-cxr/2.1.0")
    print("Embedding shape:", embeddings.shape)
    np.save("embeddings.npy", embeddings)
