import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 强制 TensorFlow 使用 CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

# 数据预处理：文本分词和编码
tokenizer = Tokenizer(num_words=5000, lower=True, split=' ')
tokenizer.fit_on_texts(X)

# 转换文本为整数序列
X_sequences = tokenizer.texts_to_sequences(X)

# 填充序列，确保每个文本长度一致
X_pad = tf.keras.preprocessing.sequence.pad_sequences(X_sequences, padding='post', maxlen=500)

# 拆分数据集，并将测试集比例设为40%
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.4, random_state=0)

# 二值化标签
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# 构建 LSTM 模型
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=X_train.shape[1]))
model.add(SpatialDropout1D(0.2))  # 防止过拟合
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))  # LSTM 层
model.add(Dense(2, activation='softmax'))  # 输出层，二分类

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
