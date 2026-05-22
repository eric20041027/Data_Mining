"""Deep EDA: look for legitimate signal to exploit.

Uses only train text+label and test text (no test labels).
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from utils import LABEL_LIST, LABEL2IDX, load_test, load_train


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def split_title_body(text: str) -> tuple[str, str]:
    """Medical abstracts usually have a title followed by '. ' then body."""
    parts = text.split(". ", 1)
    if len(parts) == 1:
        return text, ""
    return parts[0] + ".", parts[1]


def section1_textstats() -> None:
    print("=" * 70)
    print("§1 文本結構：title vs body 區隔")
    print("=" * 70)
    train = load_train()
    test = load_test()
    train["title"], train["body"] = zip(*train["condition"].map(split_title_body))
    test["title"], test["body"] = zip(*test["condition"].map(split_title_body))

    print(f"Train title 長度（字元）: median={train['title'].str.len().median():.0f}, "
          f"mean={train['title'].str.len().mean():.0f}")
    print(f"Train body 長度（字元）:  median={train['body'].str.len().median():.0f}, "
          f"mean={train['body'].str.len().mean():.0f}")
    print(f"Test  title 長度（字元）: median={test['title'].str.len().median():.0f}, "
          f"mean={test['title'].str.len().mean():.0f}")
    print(f"Test  body 長度（字元）:  median={test['body'].str.len().median():.0f}, "
          f"mean={test['body'].str.len().mean():.0f}")

    # 多少筆有效 split 出 title（body 非空）
    print(f"\nTrain 有效 title+body 切分: {(train['body'].str.len() > 0).sum()}/{len(train)}")
    print(f"Test  有效 title+body 切分: {(test['body'].str.len() > 0).sum()}/{len(test)}")


def section2_kNN_voting() -> None:
    print("\n" + "=" * 70)
    print("§2 k-NN 多數投票：對每筆 test 找最相近的 train 文本")
    print("=" * 70)
    train = load_train()
    test = load_test()

    # TF-IDF on combined corpus（為了 idf 用全 corpus 是合法的：不用 test labels）
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000,
                          sublinear_tf=True, strip_accents="unicode")
    vec.fit(pd.concat([train["condition"], test["condition"]]))
    Xtr = vec.transform(train["condition"])
    Xte = vec.transform(test["condition"])

    for k in [1, 5, 15]:
        nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
        nn.fit(Xtr)
        dist, idx = nn.kneighbors(Xte)
        sim = 1 - dist  # cosine similarity

        # 多數投票
        pred = []
        for row in idx:
            labels = train["label_idx"].iloc[row].tolist()
            pred.append(Counter(labels).most_common(1)[0][0])
        pred = np.array(pred)

        # 平均近鄰相似度
        avg_sim = sim.mean(axis=1)
        # 多少 test 樣本「非常相似於 train」（sim ≥ 0.95，幾乎重複）
        n_dup = (sim[:, 0] >= 0.95).sum()
        n_close = ((sim[:, 0] >= 0.5) & (sim[:, 0] < 0.95)).sum()
        n_far = (sim[:, 0] < 0.5).sum()
        print(f"\nk={k}:")
        print(f"  Test 樣本最近鄰相似度分布:")
        print(f"    ≥0.95 (近重複): {n_dup}")
        print(f"    0.5-0.95 (相近): {n_close}")
        print(f"    <0.5 (相距遠):   {n_far}")
        print(f"  k-NN voting 預測 label 分布: {Counter(pred)}")


def section3_class_discriminative() -> None:
    print("\n" + "=" * 70)
    print("§3 各類別的高判別力詞彙（top 15 by χ²）")
    print("=" * 70)
    from sklearn.feature_selection import chi2

    train = load_train()
    vec = TfidfVectorizer(ngram_range=(1, 1), min_df=20, max_features=20_000,
                          sublinear_tf=True, strip_accents="unicode")
    X = vec.fit_transform(train["condition"])
    feats = np.array(vec.get_feature_names_out())

    for cls_idx, cls_name in enumerate(LABEL_LIST):
        y = (train["label_idx"] == cls_idx).astype(int)
        chi, _ = chi2(X, y)
        # 只看在這類中比較多的詞
        cls_mean = X[y == 1].mean(axis=0).A1
        rest_mean = X[y == 0].mean(axis=0).A1
        mask = cls_mean > rest_mean
        chi_masked = np.where(mask, chi, -np.inf)
        top = np.argsort(chi_masked)[::-1][:15]
        print(f"\n{cls_name}:")
        print("  " + ", ".join(feats[top]))


def section4_test_OOD() -> None:
    print("\n" + "=" * 70)
    print("§4 Test 中與 train 全部都不太像的「孤立樣本」")
    print("=" * 70)
    train = load_train()
    test = load_test()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000,
                          sublinear_tf=True, strip_accents="unicode")
    vec.fit(pd.concat([train["condition"], test["condition"]]))
    Xtr = vec.transform(train["condition"])
    Xte = vec.transform(test["condition"])

    nn = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn.fit(Xtr)
    dist, _ = nn.kneighbors(Xte)
    max_sim = 1 - dist[:, 0]

    print(f"Max train similarity 分布（每筆 test）:")
    print(pd.Series(max_sim).describe().round(3))

    # 低相似度的 test 是「孤立」的
    for thresh in [0.3, 0.4, 0.5]:
        n = (max_sim < thresh).sum()
        print(f"  Test 跟 train 最大 sim < {thresh}: {n} 筆")


def section5_multilabel_per_class() -> None:
    print("\n" + "=" * 70)
    print("§5 多標籤共現：每對類別在 train 重複文本中共現的比例")
    print("=" * 70)
    train = load_train()
    train["h"] = train["condition"].map(md5)

    g = train.groupby("h")["label_idx"].apply(set)
    multi = g[g.map(len) > 1]
    print(f"Multi-label texts: {len(multi)}")

    # 5x5 共現矩陣（normalised by row sum）
    co = np.zeros((5, 5), dtype=int)
    for labs in multi:
        labs = list(labs)
        for i in range(len(labs)):
            for j in range(len(labs)):
                if i != j:
                    co[labs[i], labs[j]] += 1

    co_df = pd.DataFrame(co, index=LABEL_LIST, columns=LABEL_LIST)
    print("\n共現次數（行 = 主類別，列 = 共同出現類別）:")
    print(co_df)

    # 每行 normalise（給定主類別，看共現分布）
    print("\n共現條件機率 P(共同類 | 主類別):")
    co_pct = (co_df.T / co_df.sum(axis=1)).T.round(3)
    print(co_pct)


def section6_title_only_signal() -> None:
    print("\n" + "=" * 70)
    print("§6 標題（第一句）的判別力 — 用 TF-IDF + LogReg")
    print("=" * 70)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score

    train = load_train()
    train["title"], _ = zip(*train["condition"].map(split_title_body))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s = []
    for tr_idx, va_idx in skf.split(train, train["label_idx"]):
        tr = train.iloc[tr_idx]
        va = train.iloc[va_idx]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                              max_features=80_000)
        Xtr = vec.fit_transform(tr["title"])
        Xva = vec.transform(va["title"])
        clf = LogisticRegression(C=4.0, class_weight=None, solver="liblinear",
                                 max_iter=1000)
        clf.fit(Xtr, tr["label_idx"])
        pred = clf.predict(Xva)
        f1s.append(f1_score(va["label_idx"], pred, average="macro"))
    print(f"Title-only TF-IDF + LogReg 5-fold OOF Macro F1: {np.mean(f1s):.4f}")
    print(f"  per fold: {[f'{f:.4f}' for f in f1s]}")


def main() -> None:
    section1_textstats()
    section2_kNN_voting()
    section3_class_discriminative()
    section4_test_OOD()
    section5_multilabel_per_class()
    section6_title_only_signal()


if __name__ == "__main__":
    main()
