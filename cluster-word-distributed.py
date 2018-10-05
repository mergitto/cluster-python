# 分散表現のクラスタリング

import sys
from collections import defaultdict
from gensim.models.keyedvectors import KeyedVectors
from sklearn.cluster import KMeans


def load_model(model_file_name):
    return KeyedVectors.load(model_file_name)

def get_vocab_vectors(model):
    max_vocab = 30000
    vocab = list(model.wv.vocab.keys())[:max_vocab]
    vectors = [model.wv[word] for word in vocab]
    return vocab, vectors

def kmeans(vectors, n_clusters):
    kmeans_model = KMeans(n_clusters=n_clusters, verbose=1, random_state=42, n_jobs=-1)
    kmeans_model.fit(vectors)
    return kmeans_model

def cluster_words(kmeans_model, vocab):
    cluster_labels = kmeans_model.labels_
    cluster_to_words = defaultdict(list)
    for cluster_id, word in zip(cluster_labels, vocab):
        cluster_to_words[cluster_id].append(word)
    return cluster_to_words

def show_cluster_words(cluster_to_words):
    for words in cluster_to_words.values():
        print(words[:10])

if __name__ == '__main__':
    model = load_model(sys.argv[1])
    vocab, vectors = get_vocab_vectors(model)
    kmeans_model = kmeans(vectors, 100)
    cluster_to_words = cluster_words(kmeans_model, vocab)

    show_cluster_words(cluster_to_words)

