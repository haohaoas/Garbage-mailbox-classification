import os

def load_emails(folder_path, label):
    contents = []
    labels = []
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        with open(fpath, 'r', encoding='gbk', errors='ignore') as f:
            text = f.read()
            # 这里简单处理：取“第一个空行后”的内容作为正文
            body = text.split('\n\n', 1)[-1]
            contents.append(body)
            labels.append(label)
    return contents, labels

# 你可以按自己的实际路径修改
normal_path = '../../data/normal'
spam_path = '../../data/spam'

normal_texts, normal_labels = load_emails(normal_path, 0)
spam_texts, spam_labels = load_emails(spam_path, 1)

all_texts = normal_texts + spam_texts
all_labels = normal_labels + spam_labels