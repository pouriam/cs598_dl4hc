import pandas as pd
import tqdm
import numpy as np
import pickle
import tensorflow as tf
import pydicom
import cv2
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences


model_path = "cnn_rnn_model.h5"
tokenizer_path = "tokenizer.pkl"


model = tf.keras.models.load_model(model_path, compile=False)
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

print(model.summary())

test_df = pd.read_csv('test.tsv', sep='\t')

base_cnn = model.get_layer("densenet121")
base_cnn.trainable = False  # Freeze the encoder


def load_dicom_image(dcm_path):
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array.astype(np.float16)
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0
    img = img.astype(np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_preprocessed = preprocess_input(img_resized)  # Shape: (224, 224, 3)
    return img_preprocessed  # No batch dimension!


def beam_search_decoder(model, image, tokenizer, beam_width=4, max_len=100):

    max_seq_length = 100  # Match repeat_vector's sequence length
    start_token = tokenizer.word_index.get('<start>', 1)
    end_token = tokenizer.word_index.get('<end>', 2)
    sequences = [[ [start_token], 0.0 ]]

    # Retrieve layers by name (no indices!)
    base_cnn = model.get_layer("densenet121")
    gap_layer = model.get_layer("global_average_pooling2d")
    dense_layer = model.get_layer("dense")
    repeat_layer = model.get_layer("repeat_vector")
    embedding_layer = model.get_layer("embedding")
    concat_layer = model.get_layer("concatenate")
    lstm_layer = model.get_layer("lstm")
    output_layer = model.get_layer("dense_1")

    # Process image symbolically (no .predict() or eager tensors)
    image_batch = tf.expand_dims(tf.convert_to_tensor(image, dtype=tf.float32), axis=0)  # (1, 224, 224, 3)
    cnn_feature = base_cnn(image_batch)
    gap_output = gap_layer(cnn_feature)  # (1, 1024)
    dense_output = dense_layer(gap_output)  # (1, 256)
    cnn_feature_repeated = repeat_layer(dense_output)  # (1, 100, 256)

    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:

            padded_seq = pad_sequences([seq], maxlen=max_seq_length, padding='post')
            x_text = tf.convert_to_tensor(padded_seq, dtype=tf.int32)  # Ensure int32 for embedding
            embedded_text = embedding_layer(x_text)  # (1, 100, 256)
            embedded_text = tf.cast(embedded_text, tf.float32)  # Match dtype with cnn_feature_repeated

            # Concatenate along feature axis (axis=2)
            x_combined = concat_layer([cnn_feature_repeated, embedded_text])  # (1, 100, 512)
            lstm_out = lstm_layer(x_combined)
            preds = output_layer(lstm_out).numpy()[0]  # (vocab_size,)

            top_tokens = preds.argsort()[-beam_width:][::-1]
            for token in top_tokens:
                penalty = 1.0
                if token in seq:  # Discourage repeated tokens
                    penalty = 2.0  # Increase penalty for repetitions
                new_seq = seq + [token]
                new_score = score - np.log(preds[token] + 1e-10) * penalty
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
counted = 0
for dicom_id in tqdm.tqdm(test_df.dicom_id):
    if counted >= 150:
        break
    try:
        img = load_dicom_image(cxr_dicom_dict[dicom_id])
        report = beam_search_decoder(model, img, tokenizer, beam_width=4)
        generated_reports[dicom_id] = report
        counted += 1
    except Exception as e:
        print(e)
        skipped += 1
print(f"Total skipped files: {skipped}/{len(test_df)}")


with open('cnn_rnn.tsv', 'w', encoding='utf-8') as f:
    print('dicom_id\tgenerated', file=f)
    for dicom_id, report in sorted(generated_reports.items()):
        print(f'{dicom_id}\t{report}', file=f)
