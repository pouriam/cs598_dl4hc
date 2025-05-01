import sys
import pandas as pd
from pathlib import Path
import os
import random
import bioc
from loader import Loader
from stages import Extractor, Classifier, Aggregator

chexpert_dir = '../data/chexpert-labeler'
if chexpert_dir not in sys.path:
    sys.path.append(chexpert_dir)


def mkpath(path_str, posix=True):
    path = os.path.join(chexpert_dir, path_str)
    if posix:
        return Path(path)
    else:
        return path


df_knn = pd.read_csv('knn.tsv', sep='\t')
tempname = 'chexpert-reports-%s.csv' % random.randint(0,10**6)
print(tempname)
with open(tempname, 'w') as f:
    for text in df_knn.generated.values:
        print(text.replace(',', '\\,'), file=f)


extractor = Extractor(mkpath('phrases/mention'), mkpath('phrases/unmention'), False)
classifier = Classifier(mkpath('patterns/pre_negation_uncertainty.txt', posix=False),
                        mkpath('patterns/negation.txt'                , posix=False),
                        mkpath('patterns/post_negation_uncertainty.txt', posix=False),
                        verbose=True)

CATEGORIES = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
              "Lung Lesion", "Lung Opacity", "Edema", "Consolidation",
              "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
              "Pleural Other", "Fracture", "Support Devices"]
aggregator = Aggregator(CATEGORIES, False)


# Nearest Neighbor
loader = Loader(tempname, False)

# Load reports in place.
loader.load()
# Extract observation mentions in place.
extractor.extract(loader.collection)
# Classify mentions in place.
classifier.classify(loader.collection)
# Aggregate mentions to obtain one set of labels for each report.
labels = aggregator.aggregate(loader.collection)

print(labels.shape)
print(labels[:5])