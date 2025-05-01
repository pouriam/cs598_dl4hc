import pandas as pd
import os
import tqdm
import pickle
from report_parser import parse_report


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

    generated_reports = {}
    for pred_dicom in tqdm.tqdm(test_df.dicom_id):
        if pred_dicom not in neighbors:
            continue
        nn = neighbors[pred_dicom]
        found = False
        i = 1
        while not found:
            nearest_dicom = nn[-i]
            nearest_train_rad_id = rad_lookup[nearest_dicom]
            # print('\t', nearest_dicom)

            key = cxr_record_dict[(str(nearest_train_rad_id), nearest_dicom)]

            # Get text of report
            nearest_train_report_file = cxr_study_dict[key]
            try:
                report = parse_report(nearest_train_report_file)
                # If the report doesn't have a findings section, then get the next-closest report
                if 'findings' in report:
                    found = True
                    generated_reports[pred_dicom] = report['findings']
            except Exception as e:
                print(e)

            i += 1

    with open('knn.tsv', 'w') as f:
        print('dicom_id\tgenerated', file=f)
        for dicom_id, generated in sorted(generated_reports.items()):
            print('%s\t%s' % (dicom_id, generated), file=f)
