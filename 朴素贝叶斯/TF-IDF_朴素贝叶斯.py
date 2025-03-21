import os
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_DIR = '/Users/guyuwei/PycharmProjects/PythonProject/deep_learning/垃圾邮件识别/enron'
target_names = ['ham', 'spam']

def get_data(DATA_DIR):
    subfolders = ['enron%d' % i for i in range(1, 7)]
    data = []
    target = []
    for subfolder in subfolders:
        # spam
        spam_files = os.listdir(os.path.join(DATA_DIR, subfolder, 'spam'))
        for spam_file in spam_files:
            with open(os.path.join(DATA_DIR, subfolder, 'spam', spam_file), encoding="latin-1") as f:
                data.append(f.read())
                target.append(1)
        # ham
        ham_files = os.listdir(os.path.join(DATA_DIR, subfolder, 'ham'))
        for ham_file in ham_files:
            with open(os.path.join(DATA_DIR, subfolder, 'ham', ham_file), encoding="latin-1") as f:
                data.append(f.read())
                target.append(0)
    return data, target

# 加载数据
X, y = get_data(DATA_DIR)

# 数据预处理函数
def preprocess_text(text):
    translator = str.maketrans("", "", string.punctuation)  # 去掉标点符号
    text = text.translate(translator).lower()  # 转为小写
    text = re.sub(r"\s+", " ", text)  # 去掉多余空格
    return text

X = [preprocess_text(x) for x in X]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 使用 TF-IDF 和词袋模型
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 使用朴素贝叶斯分类器
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)


# 进行预测
y_pred = nb_classifier.predict(X_test_tfidf)

# 输出结果
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

