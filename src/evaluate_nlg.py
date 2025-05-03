import pandas as pd
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider


df_gt = pd.read_csv('../data/reference.tsv', sep='\t')
df_knn = pd.read_csv('../data/knn.tsv', sep='\t')
df_ngram = pd.read_csv('../data/3-gram.tsv', sep='\t')
df_cnn = pd.read_csv('../data/cnn_rnn.tsv', sep='\t')

df_gt = df_gt.fillna("")
df_knn = df_knn.fillna("")
df_ngram = df_ngram.fillna("")
df_cnn = df_cnn.fillna("")

references = {k: [v] for k, v in df_gt[['dicom_id', 'text']].values}
pred_knn = {k: [v] for k, v in df_knn[['dicom_id', 'generated']].values}
pred_ngram = {k: [v] for k, v in df_ngram[['dicom_id', 'generated']].values}
pred_cnn = {k: [v] for k, v in df_cnn[['dicom_id', 'generated']].values}

methods = {'knn': pred_knn, 'ngram': pred_ngram, 'nn': pred_cnn}

for method, pred in methods.items():
    # Get just the predictions from this method
    ids = set(pred.keys()) & set(references.keys())
    pred = {k: v for k, v in pred.items() if (k in ids)}
    ref = {k: v for k, v in references.items() if (k in ids)}

    bleu_scorer = Bleu(4)
    bleu_score = bleu_scorer.compute_score(ref, pred)
    print('Bleu score: \t%-10s (n=%6d):' % (method, len(ids)), bleu_score[0])


for method, pred in methods.items():
    # Get just the predictions from this method
    ids = set(pred.keys()) & set(references.keys())
    pred = {k: v for k, v in pred.items() if (k in ids)}
    ref = {k: v for k, v in references.items() if (k in ids)}

    cider_scorer = Cider(4)
    cider_score = cider_scorer.compute_score(ref, pred)
    print('Cider score: \t%-10s (n=%6d):' % (method, len(ids)), cider_score[0])
