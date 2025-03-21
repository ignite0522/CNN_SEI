import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, concatenate
from tensorflow.keras.utils import to_categorical

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
                target.append(1)  # spam为1
        # ham
        ham_files = os.listdir(os.path.join(DATA_DIR, subfolder, 'ham'))
        for ham_file in ham_files:
            with open(os.path.join(DATA_DIR, subfolder, 'ham', ham_file), encoding="latin-1") as f:
                data.append(f.read())
                target.append(0)  # ham为0
    return data, target

# 获取数据集
DATA_DIR = '/Users/guyuwei/PycharmProjects/PythonProject/deep_learning/垃圾邮件识别/enron'
X, y = get_data(DATA_DIR)

# 使用TF-IDF提取特征
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X).toarray()

# 拆分数据集，并将测试集比例设为40%
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.4, random_state=0)

# 二值化标签
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# 将输入数据的形状调整为 (样本数, 特征数, 1)，以符合 Conv1D 的输入要求
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# 使用Functional API定义模型
input_layer = Input(shape=(X_train.shape[1], 1))

# 添加卷积层和池化层，使用 padding='same'
branch1 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(input_layer)
branch2 = Conv1D(filters=128, kernel_size=4, activation='relu', padding='same')(input_layer)
branch3 = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(input_layer)

# 合并卷积层
merged = concatenate([branch1, branch2, branch3], axis=-1)

# 添加池化层
pool = MaxPooling1D(pool_size=2)(merged)

# 展平层
flatten = Flatten()(pool)

# 全连接层
dense = Dense(64, activation='relu')(flatten)

# Dropout层减少过拟合
dropout = Dropout(0.5)(dense)

# 输出层，使用softmax激活函数进行二分类
output_layer = Dense(2, activation='softmax')(dropout)

# 创建模型
model = Model(inputs=input_layer, outputs=output_layer)

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 打印模型概况
model.summary()

# 训练模型
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# 预测结果（得到类别标签）
y_pred_labels = np.argmax(model.predict(X_test), axis=1)

# 计算分类报告
y_test_labels = np.argmax(y_test, axis=1)
print(classification_report(y_test_labels, y_pred_labels))
