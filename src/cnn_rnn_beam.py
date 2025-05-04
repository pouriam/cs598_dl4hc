import pandas as pd
import tqdm
import pickle
import numpy as np
import cv2
import pydicom
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Concatenate, RepeatVector, Dropout
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from report_parser import parse_report


class BeamSearchDecoder:
    def __init__(self, model, tokenizer, max_len=100, beam_width=4):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.beam_width = beam_width
        self.word_index = tokenizer.word_index
        self.index_word = {v: k for k, v in self.word_index.items()}

    def decode(self, image_input):
        start_token = self.word_index['<start>']
        end_token = self.word_index['<end>']

        sequences = [[[start_token], 0.0]]

        for _ in range(self.max_len):
            all_candidates = []
            for seq, score in sequences:
                if seq[-1] == end_token:
                    all_candidates.append((seq, score))
                    continue

                padded_seq = pad_sequences([seq], maxlen=self.max_len, padding='post')[0]
                preds = self.model.predict([np.array([image_input]), np.array([padded_seq])], verbose=0)[0]
                top_k = np.argsort(preds)[-self.beam_width:]

                for word_idx in top_k:
                    log_prob = np.log(preds[word_idx] + 1e-12)
                    candidate = [seq + [word_idx], score - log_prob]
                    all_candidates.append(candidate)

            ordered = sorted(all_candidates, key=lambda x: x[1] / ((len(x[0]) ** 0.7 + 1e-12)))
            sequences = ordered[:self.beam_width]

            best_seq = sequences[0][0]
            decoded = ' '.join([self.index_word.get(idx, '') for idx in best_seq if idx not in [0, end_token]])
        return decoded.strip()


def load_dicom_image(dcm_path):
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6) * 255.0
    img = img.astype(np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_preprocessed = preprocess_input(img_resized)
    return img_preprocessed


def create_sequences(tokenizer, report, image_feature, max_len, teacher_forcing_prob=1.0):
    seq = tokenizer.texts_to_sequences([report])[0]
    if not seq:
        return []

    sequences = []
    input_seq = [seq[0]]
    for i in range(1, len(seq)):
        output_word = seq[i]
        padded_seq = pad_sequences([input_seq], maxlen=max_len, padding='post')[0]
        output_word = tf.keras.utils.to_categorical([output_word], num_classes=len(tokenizer.word_index) + 1)[0]
        sequences.append((image_feature, padded_seq, output_word))

        if np.random.random() > teacher_forcing_prob and len(sequences) > 0:
            preds = model.predict([np.array([image_feature]), np.array([padded_seq])], verbose=0)[0]
            pred = np.random.choice(len(preds), p=preds)
            input_seq.append(pred)
        else:
            input_seq.append(seq[i])

        input_seq = input_seq[-max_len:]

    return sequences


if __name__ == "__main__":
    # Load data and tokenizer
    with open('../data/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    train_df = pd.read_csv('../data/train.tsv', sep='\t')
    vocab_size = len(tokenizer.word_index) + 1
    max_len = 100

    # Load DICOM metadata
    cxr_record_dict = {}
    cxr_dicom_dict = {}
    with open("C:/data/physionet/physionet.org/files/mimic-cxr/2.1.0/cxr-record-list.csv") as f:
        for line in f.readlines()[1:]:
            study_id, subject_id, dicom_id, path = line.strip().split(',')
            cxr_record_dict[(subject_id, dicom_id)] = (study_id, subject_id)
            cxr_dicom_dict[dicom_id] = path

    cxr_study_dict = dict()
    with open("C:/data/physionet/physionet.org/files/mimic-cxr/2.1.0/cxr-study-list.csv",
              encoding="utf-8") as cxr_record_file:
        lines = cxr_record_file.readlines()
        for line in lines:
            items = line.strip().split(",")
            cxr_study_dict[(items[0], items[1])] = items[2]

    # Build model
    image_input = Input(shape=(224, 224, 3))
    cnn_base = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet')
    cnn_base.trainable = True
    cnn_features = cnn_base(image_input)
    gap = tf.keras.layers.GlobalAveragePooling2D()(cnn_features)
    cnn_embed = Dense(256, activation='relu')(gap)

    text_input = Input(shape=(max_len,))
    embedding = Embedding(vocab_size, 256, mask_zero=True)(text_input)
    cnn_expanded = RepeatVector(max_len)(cnn_embed)
    decoder_input = Concatenate(axis=2)([cnn_expanded, embedding])
    lstm = LSTM(512, return_sequences=False, dropout=0.3)(decoder_input)
    output = Dense(vocab_size, activation='softmax')(lstm)

    model = Model(inputs=[image_input, text_input], outputs=output)

    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=16 * (len(train_df) // 64), decay_rate=0.5)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    # Training loop with scheduled sampling
    teacher_forcing_prob = 1.0
    for epoch in range(64):
        print(f"\nEpoch {epoch + 1}/64")
    print(f"Teacher forcing probability: {teacher_forcing_prob:.2f}")

    # Regenerate sequences with current sampling probability
    X1, X2, y = [], [], []
    skipped = 0
    for pred_dicom, ref_rad in tqdm.tqdm(zip(train_df.dicom_id, train_df.study_id)):
        try:
            key = cxr_record_dict[(str(ref_rad), pred_dicom)]
            img_path = "C:/data/physionet/physionet.org/files/mimic-cxr/2.1.0/" + cxr_dicom_dict[pred_dicom]
            img = load_dicom_image(img_path)
            report = parse_report("C:/data/physionet/physionet.org/files/mimic-cxr/2.1.0/" + cxr_study_dict[key])

            if 'findings' in report:
                seqs = create_sequences(tokenizer, report['findings'], img, max_len, teacher_forcing_prob)
                for feat, seq, word in seqs:
                    X1.append(feat)
                    X2.append(seq)
                    y.append(word)
        except Exception as e:
            print(e)
            skipped += 1

            # Convert to numpy arrays
    X1 = np.array(X1)
    X2 = np.array(X2)
    y = np.array(y)

    # Train epoch
    model.fit([X1, X2], y,
              batch_size=64,
              epochs=1,
              validation_split=0.1,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])

    # Update teacher forcing probability
    if (epoch + 1) % 16 == 0:
        teacher_forcing_prob = max(0.0, teacher_forcing_prob - 0.05)

    # Save final model
    model.save('cnn_rnn_model.h5')
    print("Training complete. Use BeamSearchDecoder for inference.")

# Example inference:
# decoder = BeamSearchDecoder(model, tokenizer)
# img = load_dicom_image("path/to/dicom")
# print(decoder.decode(img))