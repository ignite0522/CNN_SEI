# CNN之垃圾邮件识别



## 数据集

这次使用的数据集是Enron-Spam数据集，Enron-Spam数据集是目前在电子邮件相关研究中使用 

最多的公开数据集之一

目录结构

![截屏2024-12-27 19.17.04](https://s2.loli.net/2025/03/21/Rko6mbhNDx1UPZI.png)



ham目录下是正常邮件，spam目录下是垃圾邮件

随便打开一个垃圾邮件看看

```
Hello, Dear Homeowner,

We have been notified that your mortgage rate is fixed at a very high interest rate.
As a result, you are overpaying, which sums up to thousands of dollars annually.

Luckily
Interest rates in the U.S. are now as low as 3.39%.
So hurry up because the forecast for rates is not looking good!

No Obligation
Free to apply
Lock in the 3.39% rate even with bad credit!
Take Action:
Click here now for details.

To unsubscribe, click here.
```

一看就是个推销的邮件

数据地址：http://www2.aueb.gr/users/ion/data/enron-spam/



## 特征提取

首先把数据从对应目录中提取出来

```py
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
```

得到的是：每个文件中的文本内容



现在得到的文本其中含有大量标点符号和特殊字符

需要去掉

为什么呢？举个例子就知道了：

`"hello."` 和 `"hello"` 会被视为不同的特征，这明显不是我们想要的



那自己定义一个函数去除它们

```py
def preprocess_text(text):
    text = re.sub(r"[$@#]+", "", text)  # 去掉 $, @, # 等特殊字符
    text = re.sub(r"[^\w\s]", "", text)  # 去掉标点符号（保留字母、数字和空格）
    text = re.sub(r"\s+", " ", text)  # 去掉多余空格
    return text
```





### 词袋模型(Bag of Word)

又称BOW

词袋模型将每个文档视为无序词汇集，只考虑每个词语的出现频率



举个简单的例子：

文档 1: "I love programming"
文档 2: "I love Python"
文档 3: "Python is great"



```
词汇表: ["I", "love", "programming", "Python", "is", "great"]
```



统计每个文档对应的单词在词汇表中对应的位置和出现的次数

```
文档 1 向量: [1, 1, 1, 0, 0, 0]
文档 2 向量: [1, 1, 0, 1, 0, 0]
文档 3 向量: [0, 0, 0, 1, 1, 1]
```

找到一个图：

![img](https://s2.loli.net/2025/03/21/t5TACuz7agn6mxR.jpg)

### TF-IDF模型

TF-IDF模型的核心思想是：**如果某个词语在一篇文档中频繁出现，但在其他文档中很少出现，则认为这个词语具有很好的类别区分能力，对文档的区分度高，因此应该给予更高的权重**

那是怎么实现这个核心思想的呢

看看公式就知道了

**TF (Term Frequency)**：表示词语在文档中出现的频率。公式如下：

![img](https://s2.loli.net/2025/03/21/y3OmD7V8nKdes4p.png)

**IDF (Inverse Document Frequency)**：表示词语在整个语料库中的逆文档频率。公式如下：

![img](https://s2.loli.net/2025/03/21/j5MpzIqtRUA7O8Y.png)

**TF-IDF**：通过将TF和IDF相乘来计算一个词语在文档中的TF-IDF值，公式如下：



![img](https://s2.loli.net/2025/03/21/brGKQvOeaXcM2pg.png)

就这样，当一个词语在本文中出现次数高，但在其他文本中出得少，即可讲这个词语判断为这篇文本的中心词语，具有代表性



在代码中我们直接用分装好的函数TfidfVectorizer即可

```py
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
```

```stop_words="english"```指定停用词"the"、"and"、"is" 等，即不考虑这些词语

```max_features=5000```指最多会选择最常见的 5000 个词来构建特征矩



**注意这里使用了transform来对测试集进行操作，这个方法仅使用已拟合的 `TfidfVectorizer` 模型来转换测试数据 `X_test`，而不会重新拟合。这样可以确保训练集和测试集的特征空间一致**



打印一下最终的数据看看

```
  (0, 4744)	0.34162197042603726
  (0, 4649)	0.409198850292271
  (0, 4517)	0.2803172283937868
  (0, 4489)	0.12533624016724074
  (0, 4463)	0.12451757447922789
  (0, 4368)	0.04317944130391292
  (0, 4329)	0.13287347442735495
  (0, 3070)	0.3274642260306118
  (0, 2869)	0.24252270084387478
  (0, 2495)	0.20723934750945244
  (0, 1950)	0.07489597080667541
  (0, 1907)	0.3027994460669858
  (0, 1625)	0.07641001110340034
  (0, 1520)	0.12160957927209229
  (0, 1291)	0.08896215192978824
  (0, 1007)	0.04896211167591468
  (0, 899)	0.13440530258676797
  (0, 852)	0.06119245421425564
  (0, 348)	0.10673266723035542
  (0, 277)	0.296177737560419
  (0, 156)	0.11369076651912009
  (0, 138)	0.10040086359203333
  (0, 69)	0.1276421038248682
  (0, 66)	0.13831297742273943
  (0, 41)	0.16244982817512751
  :	:
  (8427, 1551)	0.12798009614692601
  (8427, 1452)	0.11620946877800811
  (8427, 1030)	0.12569287727565723
  (8427, 954)	0.13662783011986893
  (8427, 870)	0.10528235266066947
  (8427, 544)	0.14601208198606558
  (8427, 478)	0.2473254498646145
  (8427, 417)	0.13058548904701373
  (8427, 355)	0.15345623443716655
  (8427, 272)	0.12401936607681793
  (8427, 109)	0.2291558524948942
  (8427, 69)	0.07133706851651772
  (8427, 37)	0.0941927906344525
  (8427, 0)	0.14910354802050016
  (8428, 4962)	0.19455214451624742
  (8428, 4844)	0.1904943142158063
  (8428, 4368)	0.06175824958635996
  (8428, 3346)	0.3425790796477523
  (8428, 3213)	0.38268146292218097
  (8428, 3051)	0.16556487536542616
  (8428, 2283)	0.36961038053994616
  (8428, 2246)	0.16378102627028024
  (8428, 1517)	0.3065864259266542
  (8428, 1007)	0.14005805641965774
  (8428, 906)	0.5947961738215157
```

可以看到这是稀疏矩阵

(行索引, 列索引)  TF-IDF值，比如

(0, 66)  0.1383指的是在第一个文本中索引为66的词的 **TF-IDF值**为0.1383

(8427, 1551）0.1279指的是在第8427个文本中索引为1551的词的 **TF-IDF值**为0.1279



## 朴素贝叶斯模型

老规矩，上代码

```py
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

X, y = get_data(DATA_DIR)

def preprocess_text(text):
    translator = str.maketrans("", "", string.punctuation)  # 去掉标点符号
    text = text.translate(translator).lower()  # 转为小写
    text = re.sub(r"\s+", " ", text)  # 去掉多余空格
    return text

X = [preprocess_text(x) for x in X]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 使用朴素贝叶斯分类器
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

y_pred = nb_classifier.predict(X_test_tfidf)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")


```



分类标准：当 P（垃圾邮件|文字内容）> P（正常邮件|文字内容）时，我们认为该邮件为垃圾邮件，但是单凭单个词而做出判断误差肯定相当大，因此我们可以将更多的词一起进行联合判断

假如我们进行判断的词有“中奖”、“免费”、“无套路”是不是垃圾邮件

![img](https://s2.loli.net/2025/03/21/A3xtwpo7LrT1Bid.jpg)

 同理，P（正常|中奖，免费，无套路）可以变为：

![img](https://s2.loli.net/2025/03/21/mD9kgoGsPHB5qrY.jpg)

因此，对P（垃圾邮件|中奖，免费，无套路）与P（正常|中奖，免费，无套路）的比较，只需要对分子进行对比。

而分子中前三个概率我们在进行TF-IDF处理时已经得到了，P(正常)就跟好算了，正常邮件数/总邮件数即可，P(垃圾邮件)同理

但是还有一个要注意的问题，如果某个词只出现在垃圾邮件中，而没有出现在正常的邮件中，这就会导致P（内容|正常）为0，从而导致整个分子都变为0，所以还要使用**拉普拉斯平滑**，说白了就是统计时要在分子上加个1

 

使用模型的话就简单了

这里就直接使用封装好了的模型就行

```py
# 使用朴素贝叶斯分类器
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# 进行预测
y_pred = nb_classifier.predict(X_test_tfidf)
```



最后的准确率有0.9810





## CNN模型

老规矩，先上代码

```py
model = Sequential()



model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

model.fit(X_train, np.array(y_train), epochs=5, batch_size=64, validation_data=(X_test, np.array(y_test)))

y_pred = (model.predict(X_test) > 0.5).astype("int32")

print(classification_report(y_test, y_pred))
```



通过 `Sequential` 模型来按顺序逐层构建神经网络

把构建好的模型打印一下

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             (None, 4996, 128)         768       
                                                                 
 max_pooling1d (MaxPooling1  (None, 2498, 128)         0         
 D)                                                              
                                                                 
 flatten (Flatten)           (None, 319744)            0         
                                                                 
 dense (Dense)               (None, 64)                20463680  
                                                                 
 dropout (Dropout)           (None, 64)                0         
                                                                 
 dense_1 (Dense)             (None, 1)                 65        
                                                                 
=================================================================

```

可以看到依次为

卷积层=>池化层=>`Flatten` 层(用于将多维输入展平为一维)

=>全连接层=>Dropout层=>输出层



接着看看

```py
model.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)))
```

这里定义了一个卷积层

这里有128个不同的卷积核，每个卷积核是一维的，这里的```kernel_size=5```表示1*5大小的一维卷积核，

**注意这里只能使用一位的卷积核，因为输入的数据中，一行代表一个文本不同列代表不同的单词，不同行之间是不同文本**，不同行之间不能做卷积



model.add(MaxPooling1D(pool_size=2))表示每次池化操作的窗口大小是 2，也就是说每两个连续的元素会进行一次池化，取出它们中的最大值，即数量减半



把完整代码给出来吧

```py
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


X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)


model = Sequential()

model.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)))

model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

model.fit(X_train, np.array(y_train), epochs=5, batch_size=64, validation_data=(X_test, np.array(y_test)))


y_pred = (model.predict(X_test) > 0.5).astype("int32")#进行二分类


print(classification_report(y_test, y_pred))
```



## TEXTCNN

到此为止上面的已经有textcnn那个味道了

那textcnn有什么独特之处呢

先看一张图

![在这里插入图片描述](https://s2.loli.net/2025/03/21/zcFhON2MgEmQUXW.png)

可以看到和我之前构建的cnn模型只在卷积核这里有所区分

这里使用了分别使用了kernel_size为2，3，4

注意这里还是使用的1维卷积核



**最开始看错了**

**这幅图是每一列代表一个文本，每行代表每个文本的单词**



说一下整个过程

这个卷积和池化操作和我们平时横着看的思维不一样，在这里，卷积核是向下移动的

第一个kernel_size为4的卷积核向下一次滑动一个，总共滑动4次，得到4*1的矩阵

第二个kernel_size为3的卷积核向下一次滑动一个，总共滑动5次，得到5*1的矩阵

第三个kernel_size为2的卷积核向下一次滑动一个，总共滑动6次，得到6*1的矩阵

然后见得到的矩阵池化，再进行拼接，最后输入到全连接层



先上代码

```py
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

DATA_DIR = '/Users/guyuwei/PycharmProjects/PythonProject/deep_learning/垃圾邮件识别/enron'
X, y = get_data(DATA_DIR)

tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.4, random_state=0)

# 二值化标签
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# 将输入数据的形状调整为 (样本数, 特征数, 1)，以符合 Conv1D 的输入要求
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# 使用Functional API定义模型
input_layer = Input(shape=(X_train.shape[1], 1))

branch1 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(input_layer)
branch2 = Conv1D(filters=128, kernel_size=4, activation='relu', padding='same')(input_layer)
branch3 = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(input_layer)

merged = concatenate([branch1, branch2, branch3], axis=-1)

pool = MaxPooling1D(pool_size=2)(merged)

flatten = Flatten()(pool)

dense = Dense(64, activation='relu')(flatten)

dropout = Dropout(0.5)(dense)

# 输出层，使用softmax激活函数进行二分类
output_layer = Dense(2, activation='softmax')(dropout)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 打印模型概况
model.summary()

model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

y_pred_labels = np.argmax(model.predict(X_test), axis=1)


y_test_labels = np.argmax(y_test, axis=1)
print(classification_report(y_test_labels, y_pred_labels))

```



可以看到主要变化就是

```py
branch1 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(input_layer)
branch2 = Conv1D(filters=128, kernel_size=4, activation='relu', padding='same')(input_layer)
branch3 = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(input_layer)
```

使用了三种不同的卷积核

那这样做的目的是干什么呢？

之前我们使用的是kernel_size=5

现在多加了3和4，现在不仅可以考虑到范围为5以内的单词，还可以考虑到范围分别为3，4以内的单词，这样考虑到范围变得丰富了，不仅可以实现之前的捕捉到跨越上下文的信息，还可以实现获得更丰富的特征信息，实现了多范围考虑



最后还要使用concatenate把不同卷积，再池化过后的值连接起来

更详细的解释看：https://blog.csdn.net/GFDGFHSDS/article/details/105295247



## 对比

在做这个垃圾邮件识别的项目中，数据量还算大吧

使用机器学习朴素贝叶斯的正确率为98.1%



使用CNN有所提高，到了99%

![image-20241227231323316](https://s2.loli.net/2025/03/21/rxuWqAYFMkl7LVX.png)



那这其中是哪里增加了正确率呢

从CNN卷积核这里分析

```py
model.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)))
```

卷积核考虑了5列，即一个文本中的5个单词，CNN可以捕获文本中隐藏的语义关系，能够处理更加复杂和非结构化的数据，从而提升分类性能



那紧接着又会有疑问了，我们这里可是使用的TF-IDF哦，**TF-IDF 计算出的结果不包含顺序信息，因此在本质上没有“前后考虑”**，即它是基于每个词的出现频率和逆文档频率（Inverse Document Frequency）来表示文本的特征，而不考虑文本中的词序列

理论上TF-IDF和CNN结合起来并不会有什么用

但我们这里他就是提高了1%，可能会有一点点关系吧，这就是学人工智能有意思的地方，不知道为啥，反正这样做就是提高了正确率，哈哈！！



最后再讲讲一次有趣的尝试吧

既然说使用TF-IDF去结合CNN没用，那我直接将单词转化成对应的向量总行了吧



我试着不使用TF-IDF模型去提取特征

就只将文本中对应的单词转换成对应的向量矩阵

```py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, concatenate
from tensorflow.keras.utils import to_categorical
import tensorflow as tf


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

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.4, random_state=0)

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# 使用Functional API定义模型
input_layer = Input(shape=(X_train.shape[1], 1))

branch1 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(input_layer)
branch2 = Conv1D(filters=128, kernel_size=4, activation='relu', padding='same')(input_layer)
branch3 = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(input_layer)

merged = concatenate([branch1, branch2, branch3], axis=-1)

pool = MaxPooling1D(pool_size=2)(merged)

flatten = Flatten()(pool)

dense = Dense(64, activation='relu')(flatten)

dropout = Dropout(0.5)(dense)

output_layer = Dense(2, activation='softmax')(dropout)


model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

class_weight = {0: 1., 1: 2.}  # 增大垃圾邮件（类别1）的权重
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), class_weight=class_weight)

y_pred_labels = np.argmax(model.predict(X_test), axis=1)

# 计算分类报告
y_test_labels = np.argmax(y_test, axis=1)
print(classification_report(y_test_labels, y_pred_labels))
```

结果已经想到了，正确率好低

这还是完善了的结果，最开始没有加这个

```py
class_weight = {0: 1., 1: 2.}  # 增大垃圾邮件（类别1）的权重
```

垃圾邮件的正确率为0，之后增大了垃圾邮件（类别1）的权重，就像下图这样了

![截屏2024-12-27 23.09.19](https://s2.loli.net/2025/03/21/HsndEoXZcDFLp6a.png)



## 总结

就到这里吧，应该是不能直接简单的将单词映射成向量去考虑，我认为造成这样低低正确率可能是因为虽然我们考虑到了前后单词的关系，但有些单词前后的关系是没有意义的，又或者是前后考虑时，中间其实是有标点的，考虑到的单词根本没有关联性

接下来打算学一下word2vec，可能会有所突破
