# 自然言語のデータを使ってクラスタリングを行う

## 文書のクラスタリング
分かち書きされた文書を用意
```
python n_clusters cluster.py wakati.file.name
```
n_clusters…クラスタ数

## 分散表現のクラスタリング
分散表現モデルを準備(word2vec, Glove, fastTextなどを利用)
```
python cluster-word-distributed.py word2vec.model.name
```

