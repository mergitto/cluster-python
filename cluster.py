# 分かち書きされた文書のクラスタリング

import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def wakati_to_docs(wakati_file_name):
    df = pd.read_csv(wakati_file_name, header=None)
    docs = np.array(df).reshape(-1,)
    return docs

def get_vecs(docs):
    vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b')
    vecs = vectorizer.fit_transform(docs)
    return vecs

def get_docs_cluster(wakati_file_name, n_clusters):
    docs = wakati_to_docs(wakati_file_name)
    vecs = get_vecs(docs)
    clusters = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(vecs)
    return docs, clusters

def show_cluster_doc(docs, clusters):
    for doc, cls in zip(docs, clusters):
        print(cls, doc)

def write_csv(docs, clusters):
    pd.DataFrame({
        'cluster_number': clusters,
        'documents': docs,
    }).to_csv(sys.argv[3], index=None)

if __name__ == '__main__':
    n_clusters = int(sys.argv[1])
    docs, clusters = get_docs_cluster(sys.argv[2], n_clusters)
    write_csv(docs, clusters)
    #show_cluster_doc(docs, clusters)
