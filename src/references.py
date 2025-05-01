import pandas as pd
from report_parser import parse_report
import tqdm


test_df = pd.read_csv('../data/test.tsv', sep='\t')

cxr_record_dict = dict()
with open("cxr-record-list.csv", encoding="utf-8") as cxr_record_file:
    lines = cxr_record_file.readlines()
    for line in lines:
        items = line.strip().split(",")
        cxr_record_dict[(items[1], items[2])] = (items[0], items[1])

cxr_study_dict = dict()
with open("cxr-study-list.csv", encoding="utf-8") as cxr_record_file:
    lines = cxr_record_file.readlines()
    for line in lines:
        items = line.strip().split(",")
        cxr_study_dict[(items[0], items[1])] = items[2]

ref_reports = {}
for pred_dicom, ref_rad in tqdm.tqdm(zip(test_df.dicom_id, test_df.study_id)):

    key = cxr_record_dict[(str(ref_rad), pred_dicom)]

    try:
        report = parse_report(cxr_study_dict[key])
        ref_reports[pred_dicom] = report['findings']
    except Exception as e:
        pass

with open('reference.tsv', 'w') as f:
    print('dicom_id\ttext', file=f)
    for dicom_id, generated in sorted(ref_reports.items()):
        print('%s\t%s' % (dicom_id, generated), file=f)
