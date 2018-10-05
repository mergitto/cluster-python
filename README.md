# 自然言語のデータを使ってクラスタリングを行う

## 文書のクラスタリング
分かち書きされた文書を用意
```
python cluster.py wakati.file.name
```

## 分散表現のクラスタリング
分散表現モデルを準備(word2vec, Glove, fastTextなどを利用)
```
python cluster-word-distributed.py word2vec.model.name
```

