import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, concatenate
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

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

# 使用TextVectorization进行文本编码
max_document_length = 500  # 假设最大文档长度为500个单词
vectorizer = tf.keras.layers.TextVectorization(max_tokens=5000, output_mode='int', output_sequence_length=max_document_length)

# 适配（学习词汇表）
vectorizer.adapt(X)

# 转换文本数据
X_encoded = vectorizer(X)

# 将 X_encoded 转换为 NumPy 数组
X_encoded = X_encoded.numpy()

# 划分数据集，并将测试集比例设为40%
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.4, random_state=0)

# 二值化标签
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# 将输入数据的形状调整为 (样本数, 特征数, 1)，以符合 Conv1D 的输入要求
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# 使用Functional API定义模型
input_layer = Input(shape=(X_train.shape[1], 1))

# 添加卷积层和池化层，使用 padding='same'
branch1 = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(input_layer)



# 添加池化层
pool = MaxPooling1D(pool_size=2)(branch1)

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

class_weight = {0: 1., 1: 2.}  # 增大垃圾邮件（类别1）的权重
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), class_weight=class_weight)


# 预测结果（得到类别标签）
y_pred_labels = np.argmax(model.predict(X_test), axis=1)

# 计算分类报告
y_test_labels = np.argmax(y_test, axis=1)
print(classification_report(y_test_labels, y_pred_labels))
