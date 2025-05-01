import pandas as pd
import numpy as np
import os
import tqdm
import pickle
from report_parser import parse_report
from collections import Counter, defaultdict


START = '<START>'
END = '<END>'


def fit(dicom_ids, rad_lookup, n=3):
    # Language model
    LM = defaultdict(Counter)

    for dicom_id in dicom_ids:

        rad_id = rad_lookup[dicom_id]

        key = cxr_record_dict[(str(rad_id), dicom_id)]

        # Get text of report
        nearest_train_report_file = cxr_study_dict[key]
        try:
            parsed_report = parse_report(nearest_train_report_file)

            if 'findings' in parsed_report:
                toks = parsed_report['findings'].replace('.', ' . ').split()
                padded_toks = [START for _ in range(n - 1)] + toks + [END]
                for i in range(len(padded_toks) - n + 1):
                    context = tuple(padded_toks[i:i + n - 1])
                    target = padded_toks[i + n - 1]

                    # TODO: get similarities
                    # sim = sim_score(img1,img2)
                    sim = 1

                    LM[context][target] += sim
        except Exception as e:
            print(e)
    return LM


def sample(seq_so_far):
    #print(seq_so_far)
    last = tuple(seq_so_far[-n+1:])
    words,P = list(zip(*LM[last].items()))
    P = np.array(P) / sum(P)
    choice = np.random.choice(words, p=P)
    return choice
    #y = clf.predict(x)[0]
    #next_word = y_vect.translate(y)
    #return next_word


if __name__ == "__main__":

    with open('../data/camera_ready_top100.pkl', 'rb') as f:
        neighbors = pickle.load(f)

    print("neighbors len: ", len(neighbors))

    data_dir = "C:/Users/pouria/Documents/Illinois/cs598-dl-for-healthcare/Project/cs598_dl4hc/data/"
    reports_path = "C:/data/physionet/physionet.org/files/mimic-cxr/2.1.0"
    cxr_record_dict = dict()
    with open(f"{reports_path}/cxr-record-list.csv", encoding="utf-8") as cxr_record_file:
        lines = cxr_record_file.readlines()
        for line in lines:
            items = line.strip().split(",")
            cxr_record_dict[(items[1], items[2])] = (items[0], items[1])

    cxr_study_dict = dict()
    with open(f"{reports_path}/cxr-study-list.csv", encoding="utf-8") as cxr_record_file:
        lines = cxr_record_file.readlines()
        for line in lines:
            items = line.strip().split(",")
            cxr_study_dict[(items[0], items[1])] = items[2]

    train_df = pd.read_csv(os.path.join(data_dir, 'train.tsv'), sep='\t')
    test_df = pd.read_csv(os.path.join(data_dir, 'test.tsv'), sep='\t')

    print("train data shape:", train_df.shape)

    print("test data shape:", test_df.shape)

    # Map each dicom to its study_id
    rad_lookup = dict(train_df[['dicom_id', 'study_id']].values)

    n = 3

    generated_reports = {}
    for pred_dicom in tqdm.tqdm(test_df.dicom_id):

        # Build ngram model from the neighbors
        if pred_dicom not in neighbors:
            continue

        nn = neighbors[pred_dicom]
        LM = fit(nn, rad_lookup, n=n)

        # get generated report by sampling from the ngram model
        #   (i.e. select next word with probability that it follows given (n-1) words)
        generated_toks = [START for _ in range(n - 1)]
        current = generated_toks[-1]
        while current != END and len(generated_toks) < 100:
            next_word = sample(generated_toks)
            # print(next_word)
            generated_toks.append(next_word)
            current = next_word
            # break
        generated_toks = generated_toks[n - 1:]
        if generated_toks[-1] == END: generated_toks[:-1]

        # Store generated sentence
        g_toks = ' '.join(generated_toks)
        generated_reports[pred_dicom] = g_toks

    with open(f'{n}-gram.tsv', 'w') as f:
        print('dicom_id\tgenerated', file=f)
        for dicom_id,generated in sorted(generated_reports.items()):
            print('%s\t%s' % (dicom_id,generated), file=f)
