import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
import pydicom
import random
from collections import defaultdict
import matplotlib.pyplot as plt


# Path to DICOM files directory
# dicom_directory = "C:/Users/pouria/Documents/Illinois/cs598-dl-for-healthcare/Project/cs598_dl4hc/data"
dicom_directory = "C:/data/physionet/physionet.org/files/mimic-cxr/2.1.0"

densenet_model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, pooling='avg')


def load_dicom_image(dcm_path):
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0
    img = img.astype(np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_preprocessed = preprocess_input(np.expand_dims(img_resized, axis=0))
    return img_preprocessed


def get_dicom_embeddings(dicom_dir):
    embeddings = {}
    for (dirpath, dirnames, filenames) in os.walk(dicom_dir):
        for filename in filenames:
            if filename.lower().endswith('.dcm'):
                try:
                    dcm_path = os.path.join(dirpath, filename)

                    img_preprocessed = load_dicom_image(dcm_path)

                    # Generate embedding
                    embedding = densenet_model.predict(img_preprocessed, verbose=0)
                    embeddings[filename.replace(".dcm", "")] = embedding.flatten()
                except Exception as e:
                    print(e, filename)

    return embeddings


def extract_view_positions():
    view_positions = {}
    # Iterate through each DICOM file to extract ViewPosition metadata
    for (dirpath, dirnames, filenames) in os.walk(dicom_directory):
        for filename in filenames:
            if filename.lower().endswith('.dcm'):
                try:
                    dcm_path = os.path.join(dirpath, filename)
                    dcm_data = pydicom.dcmread(dcm_path, stop_before_pixels=True)

                    # Check if ViewPosition metadata is present
                    view_position = getattr(dcm_data, 'ViewPosition', 'Unknown')
                    view_positions[filename.replace(".dcm", "")] = view_position
                except Exception as e:
                    print(e, filename)

    return view_positions


def display_dcm_files(dicom_dir):
    for dcm_file in os.listdir(dicom_dir):
        dcm_path = os.path.join(dicom_dir, dcm_file)
        dcm_data = pydicom.dcmread(dcm_path)

        # Get ViewPosition (e.g., AP, PA)
        view_position = getattr(dcm_data, 'ViewPosition', 'Unknown')

        img = dcm_data.pixel_array

        plt.imshow(img, cmap='gray')
        plt.title(f'{dcm_file}\nView Position: {view_position}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def generate_data():
    data = defaultdict(list)

    df_records = pd.read_csv(dicom_directory + "/cxr-record-list.csv")
    df_records.fillna('', inplace=True)

    for _, row in df_records.iterrows():
        dicom_file_path = os.path.join(dicom_directory, row['path'])
        if row['dicom_id'] in view_positions:
            view_position = view_positions[row['dicom_id']]
            print(row['dicom_id'], view_position)
            if view_position != 'AP':
                continue
            data[row['dicom_id']].append(dicom_file_path)

    return data


def create_data_frame():
    ROOT_DIR = "C:/data/physionet/physionet.org/files/mimic-cxr"

    csv_file_path = f'{ROOT_DIR}/2.1.0/cxr-record-list.csv'
    df = pd.read_csv(csv_file_path, sep=',', header=0)
    df.reset_index()

    print(df.shape)  # Original -> (473057, 4)
    print(df.head())

    print('unique subjects: %6d' % len(set(df['subject_id'])))  # Original -> unique subjects:  63478
    print('unique     rads: %6d' % len(set(df['study_id'])))  # Original -> unique     rads: 206563
    print('unique   dicoms: %6d' % len(set(df['dicom_id'])))  # Original -> unique   dicoms: 473057

    return df


if __name__ == "__main__":
    df = create_data_frame()
    densenet_vecs = get_dicom_embeddings(dicom_directory)
    df = df[df.dicom_id.isin(densenet_vecs.keys())]
    view_positions = extract_view_positions()

    generated_data = generate_data()

    dicom_ids = list(generated_data.keys())
    random.shuffle(dicom_ids)

    n = len(dicom_ids)
    split_ind = int(0.7 * n)
    train_ids = dicom_ids[:split_ind]
    test_ids = dicom_ids[split_ind:]

    print('train:', len(train_ids))
    print('test: ', len(test_ids))

    df[df.dicom_id.isin(train_ids)].to_csv('train.tsv', sep='\t')
    df[df.dicom_id.isin(test_ids)].to_csv('test.tsv', sep='\t')
