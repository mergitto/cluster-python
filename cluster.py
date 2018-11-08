# 分かち書きされた文書のクラスタリング

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def load_pickle():
    import pickle
    with open("./advice_2_tfidf.pickle", 'rb') as f:
        return pickle.load(f)

def dump_pickle(pickle_data, save_file_name):
    import pickle
    with open(save_file_name, 'wb') as f:
        pickle.dump(pickle_data, f)

def wakati_to_docs(wakati_documents):
    docs = np.array(wakati_documents).reshape(-1,)
    return docs

def get_vecs(docs):
    vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b')
    vecs = vectorizer.fit_transform(docs)
    return vecs

def pluck_wakati_space_advice(reports):
    return [report["advice_divide_mecab_space"] for report in reports.values()]

def get_docs_cluster(reports, n_clusters):
    wakati_documents = pluck_wakati_space_advice(reports)
    docs = wakati_to_docs(wakati_documents)
    vecs = get_vecs(docs)
    clusters = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(vecs)
    return docs, clusters

def show_cluster_doc(docs, clusters):
    for doc, cls in zip(docs, clusters):
        print(cls, doc)

def write_csv(docs, clusters, save_file_name):
    pd.DataFrame({
        'cluster_number': clusters,
        'documents': docs,
    }).to_csv(save_file_name, index=None)

if __name__ == '__main__':
    n_clusters = 10
    reports = load_pickle()
    docs, clusters = get_docs_cluster(reports, n_clusters)

    for index in reports:
        reports[index]["cluster"] = clusters[index]

    dump_pickle(reports, "advice_2_cluster.pickle")

    #write_csv(docs, clusters, "cluster.csv")
    #show_cluster_doc(docs, clusters)
