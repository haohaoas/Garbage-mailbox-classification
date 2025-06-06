import re
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
from src.features.dataPreprocesing import all_texts

# 读取你的停用词表
with open('../../data/stopwords.txt', encoding='GBK') as f:
    stopwords = [line.strip() for line in f if line.strip()]


def clean_tokens(tokens):
    filtered = []
    for word in tokens:
        if re.fullmatch(r'\d+', word):  # 纯数字
            continue
        if re.fullmatch(r'\W+', word):  # 纯符号
            continue
        if len(word.strip()) <= 1:      # 单字
            continue
        if re.fullmatch(r'[\d.]+[a-zA-Z]*', word):  # 数字加字母
            continue
        if re.fullmatch(r'[a-zA-Z]+\d*', word) and len(word) <= 3:  # 短英文+数字
            continue
        if word.isalpha() and len(word) <= 2:  # 1~2位纯英文
            continue
        filtered.append(word)
    return filtered

def chinese_tokenizer(texts):
    return [' '.join(clean_tokens(jieba.cut(text))) for text in texts]

all_texts_cut = chinese_tokenizer(all_texts)

vectorizer = TfidfVectorizer(stop_words=stopwords, max_features=5000)
X = vectorizer.fit_transform(all_texts_cut)

print("特征维度：", X.shape)
print("前10个特征词：", vectorizer.get_feature_names_out()[:50])