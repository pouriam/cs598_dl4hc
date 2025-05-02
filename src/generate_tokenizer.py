import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer


def create_tokenizer():
    # Load your training reports
    train_df = pd.read_csv('../data/train.tsv', sep='\t')

    dicom_ids = train_df['dicom_id'].fillna("").tolist()
    cxr_record_dict = dict()
    with open("C:/data/physionet/physionet.org/files/mimic-cxr/2.1.0/cxr-record-list.csv", encoding="utf-8") as cxr_record_file:
        lines = cxr_record_file.readlines()
        for line in lines:
            items = line.strip().split(",")
            cxr_record_dict[(items[2])] = (items[0], items[1])

    cxr_study_dict = dict()
    with open("C:/data/physionet/physionet.org/files/mimic-cxr/2.1.0/cxr-study-list.csv", encoding="utf-8") as cxr_record_file:
        lines = cxr_record_file.readlines()
        for line in lines:
            items = line.strip().split(",")
            cxr_study_dict[(items[0], items[1])] = items[2]

    # Create tokenizer
    vocab_size = 5000
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<unk>')
    texts = []
    skipped = 0
    for dicom_id in dicom_ids:
        file_path = cxr_study_dict[cxr_record_dict[dicom_id]]
        try:
            with open("C:/data/physionet/physionet.org/files/mimic-cxr/2.1.0/" + file_path) as f:
                texts.append(f.read())
        except Exception as e:
            print(e)
            skipped += 1
    print(f"Total skipped files: {skipped}/{len(dicom_ids)}")
    tokenizer.fit_on_texts(texts)

    # Save tokenizer
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)


if __name__ == "__main__":
    create_tokenizer()
