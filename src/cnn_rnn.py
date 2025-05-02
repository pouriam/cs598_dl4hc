import pandas as pd
import tqdm
import numpy as np
import pickle
import tensorflow as tf
import pydicom
import cv2
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Add


model_path = "../data/cnn_rnn_model.h5"
tokenizer_path = "../data/tokenizer.pkl"


model = tf.keras.models.load_model(model_path, compile=False)
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

# === Load test metadata ===
test_df = pd.read_csv('../data/test.tsv', sep='\t')

base_cnn = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, pooling='avg')
base_cnn.trainable = False  # Freeze the encoder


# def preprocess_image(dcm_path):
#     dcm = pydicom.dcmread(dcm_path)
#     img = dcm.pixel_array.astype(np.float32)
#     img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0
#     img = img.astype(np.uint8)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#     img_resized = cv2.resize(img_rgb, (224, 224))
#     img_preprocessed = preprocess_input(np.expand_dims(img_resized, axis=0))
#     return img_preprocessed


def load_dicom_image(dcm_path):
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0
    img = img.astype(np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_preprocessed = preprocess_input(img_resized)  # Shape: (224, 224, 3)
    return img_preprocessed  # No batch dimension!


def beam_search_decoder(model, image, tokenizer, beam_width=4, max_len=30):
    max_seq_length = 30
    start_token = tokenizer.word_index.get('<start>', 1)
    end_token = tokenizer.word_index.get('<end>', 2)

    sequences = [[ [start_token], 0.0 ]]

    cnn_feature = base_cnn.predict(np.expand_dims(image, axis=0))
    cnn_feature = tf.keras.layers.Dense(256, activation='relu')(cnn_feature)

    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            padded_seq = tf.keras.preprocessing.sequence.pad_sequences([seq], maxlen=max_seq_length)
            x_text = tf.convert_to_tensor(padded_seq)
            embedded_text = model.get_layer(index=4)(x_text) # shape: (1, max_seq_length, 256)

            # Expand CNN feature for concat
            img_embed = tf.expand_dims(cnn_feature, 1)
            x_combined = tf.concat([img_embed, embedded_text[:, :-1, :]], axis=1)
            # repeated_feature = tf.repeat(cnn_feature[:, np.newaxis, :], max_seq_length, axis=1)
            # x_combined = tf.concat([repeated_feature, embedded_text], axis=-1)

            lstm_out = model.get_layer(index=6)(x_combined)
            preds = model.get_layer(index=7)(lstm_out).numpy()[0, len(seq)-1]  # get next-token probs

            top_tokens = preds.argsort()[-beam_width:][::-1]
            for token in top_tokens:
                new_seq = seq + [token]
                new_score = score - np.log(preds[token] + 1e-10)
                all_candidates.append([new_seq, new_score])

        sequences = sorted(all_candidates, key=lambda tup: tup[1])[:beam_width]
        if all(seq[-1] == end_token for seq, _ in sequences):
            break

    best_seq = sequences[0][0]
    decoded = tokenizer.sequences_to_texts([best_seq])[0]
    return decoded.replace('<start>', '').replace('<end>', '').strip()


cxr_dicom_dict = dict()
with open("cxr-record-list.csv", encoding="utf-8") as cxr_record_file:
    lines = cxr_record_file.readlines()
    for line in lines:
        items = line.strip().split(",")
        cxr_dicom_dict[items[2]] = items[3]

generated_reports = {}
skipped = 0
for dicom_id in tqdm.tqdm(test_df.dicom_id):
    try:
        img = load_dicom_image(cxr_dicom_dict[dicom_id])
        report = beam_search_decoder(model, img, tokenizer, beam_width=4)
        generated_reports[dicom_id] = report
    except Exception as e:
        skipped += 1
print(f"Total skipped files: {skipped}/{len(test_df)}")


with open('cnn_rnn.tsv', 'w', encoding='utf-8') as f:
    print('dicom_id\tgenerated', file=f)
    for dicom_id, report in sorted(generated_reports.items()):
        print(f'{dicom_id}\t{report}', file=f)
