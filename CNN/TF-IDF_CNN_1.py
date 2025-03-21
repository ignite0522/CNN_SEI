import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# 获取数据
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

# 获取数据集
DATA_DIR = '/Users/guyuwei/PycharmProjects/PythonProject/deep_learning/垃圾邮件识别/enron'
X, y = get_data(DATA_DIR)

# 使用TF-IDF提取特征
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X).toarray()

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# 构建CNN模型
model = Sequential()

# 添加卷积层
model.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)))

# 添加池化层
model.add(MaxPooling1D(pool_size=2))

# 展平层，将多维输入一维化
model.add(Flatten())

# 全连接层
model.add(Dense(64, activation='relu'))

# Dropout层以减少过拟合
model.add(Dropout(0.5))

# 输出层，2个神经元，分别对应ham和spam
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 打印模型概况
model.summary()

# 调整数据形状，增加一个维度以适应Conv1D层
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

model.fit(X_train, np.array(y_train), epochs=5, batch_size=64, validation_data=(X_test, np.array(y_test)))


y_pred = (model.predict(X_test) > 0.5).astype("int32")


print(classification_report(y_test, y_pred))
