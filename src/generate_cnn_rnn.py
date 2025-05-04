import pandas as pd
import tqdm
import pickle
from report_parser import parse_report
import numpy as np
import cv2
import pydicom
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Concatenate, RepeatVector
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_dicom_image(dcm_path):
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0
    img = img.astype(np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_preprocessed = preprocess_input(img_resized)  # Shape: (224, 224, 3)
    return img_preprocessed  # No batch dimension!


def create_sequences(tokenizer, report, image_feature, max_len):
    seq = tokenizer.texts_to_sequences([report])[0]
    sequences = []
    for i in range(1, len(seq)):
        input_seq = seq[:i]
        output_word = seq[i]
        input_seq = pad_sequences([input_seq], maxlen=max_len)[0]
        output_word = to_categorical([output_word], num_classes=vocab_size)[0]
        sequences.append((image_feature, input_seq, output_word))
    return sequences


if __name__ == "__main__":

    with open('../data/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    train_df = pd.read_csv('../data/train.tsv', sep='\t')

    vocab_size = len(tokenizer.word_index) + 1
    max_len = 100  # adjust based on your data

    cxr_record_dict = dict()
    cxr_dicom_dict = dict()
    with open("C:/data/physionet/physionet.org/files/mimic-cxr/2.1.0/cxr-record-list.csv", encoding="utf-8") as cxr_record_file:
        lines = cxr_record_file.readlines()
        for line in lines:
            items = line.strip().split(",")
            cxr_record_dict[(items[1], items[2])] = (items[0], items[1])
            cxr_dicom_dict[items[2]] = items[3]

    cxr_study_dict = dict()
    with open("C:/data/physionet/physionet.org/files/mimic-cxr/2.1.0/cxr-study-list.csv", encoding="utf-8") as cxr_record_file:
        lines = cxr_record_file.readlines()
        for line in lines:
            items = line.strip().split(",")
            cxr_study_dict[(items[0], items[1])] = items[2]

    X1, X2, y = [], [], []
    skipped = 0
    counted = 0
    for pred_dicom, ref_rad in tqdm.tqdm(zip(train_df.dicom_id, train_df.study_id)):
        if counted >= 7000:
            break
        try:
            key = cxr_record_dict[(str(ref_rad), pred_dicom)]
            img = load_dicom_image("C:/data/physionet/physionet.org/files/mimic-cxr/2.1.0/" + cxr_dicom_dict[pred_dicom])
            report = parse_report("C:/data/physionet/physionet.org/files/mimic-cxr/2.1.0/" + cxr_study_dict[key])
            # If the report doesn't have a findings section, then get the next-closest report
            if 'findings' in report:
                rep = report['findings']
                for i_feat, i_seq, i_word in create_sequences(tokenizer, rep, img, max_len):
                    X1.append(i_feat)
                    X2.append(i_seq)
                    y.append(i_word)
            counted += 1
        except Exception as e:
            skipped += 1

    print(f"Total skipped files: {skipped}/{len(train_df)}")
    X1 = np.array(X1)
    X2 = np.array(X2)
    y = np.array(y)

    print("X1 shape:", X1.shape)
    print("X2 shape:", X2.shape)
    print("y shape:", y.shape)

    # CNN encoder
    image_input = Input(shape=(224, 224, 3))
    cnn_base = tf.keras.applications.DenseNet121(include_top=False, pooling='avg', weights='imagenet')
    cnn_base.trainable = False
    cnn_features = cnn_base(image_input)
    cnn_embed = Dense(256, activation='relu')(cnn_features)

    # Text decoder
    text_input = Input(shape=(max_len,))
    cnn_expanded = RepeatVector(max_len)(cnn_embed)  # (batch, 1, 256)
    # print("cnn_expanded shape:", cnn_expanded.shape)
    embedding_layer = Embedding(vocab_size, 256, mask_zero=True)
    embedding = embedding_layer(text_input)
    # print("Embedding shape:", embedding.shape)

    decoder_input = Concatenate(axis=2)([cnn_expanded, embedding])  # (batch, 1+max_len, 256)
    # print("decoder_input shape:", decoder_input.shape)
    lstm = LSTM(512, return_sequences=False)(decoder_input)
    output = Dense(vocab_size, activation='softmax')(lstm)

    model = Model(inputs=[image_input, text_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()

    model.fit([X1, X2], y, epochs=10, batch_size=64, validation_split=0.1)
    model.save('cnn_rnn_model.h5')
