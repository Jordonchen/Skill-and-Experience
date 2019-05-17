#coding:utf-8
# Copyright: Jordonchen 2019/5/17
import numpy as np
from sklearn.datasets import load_iris   #加载鸢尾花数据集
# from sklearn.cross_validation import train_test_split   这边原始方法引入交叉验证方法会报错，换一种导入
from sklearn.model_selection import train_test_split
from collections import defaultdict
from operator import itemgetter

dataset = load_iris()
X = dataset.data
y = dataset.target
print(dataset.DESCR)  #输出数据详情
n_samples, n_features = X.shape
print("There are {0} samples and {1} features".format(n_samples, n_features))

attribute_means = X.mean(axis = 0)  #取每一列（axis=0)均值，也就是sepal length、sepel width、petal length、petal width的均值
assert attribute_means.shape == (n_features,)
X_d = np.array(X >= attribute_means, dtype = 'int')
print(X_d)  #此数据是离散化后的数据集    也就是高于平均值的取1，低于平均值的取0

"""
我们把数据集分一下，分为训练集和测试集。sklearn提供了一个数据集切分的函数train_test_split（）， \
在sklearn.cross_validation中(报错），改成sklearn.model_selection中。
"""
random_state = 14  #设置随机切分比例
X_train, X_test, y_train, y_test = train_test_split(X_d, y, random_state = random_state)
print("There are {0} training samples".format(y_train.shape))
print("There are {0} testing samples".format(y_test.shape))

"""
实现OneR算法的思路：首先遍历每一个特征的每一个取值，对于每一个特征值，统计它在每个类别中出现的次数，\
找出它出现最多的次数所对应的类别，并且统计它在其它类别中出现的次数(用来计算错误率）。\
对于鸢尾花数据集，有四个特征，三种类别。对每个特征值value，上面已经做了离散化，有两个取值（0或者1）。三种类别 \
分别用0、1、2表示y_true。feature_index表示特征索引值（0、1、2、3）对应四个花的特征,X为数据集。
"""
def train_feature_value(X, y_true, feature_index, value):
    class_counts = defaultdict(int)
    for sample, y in zip(X, y_true):  #zip()函数作用是用于将可迭代的对象作为参数，\
        # 将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象.
        if sample[feature_index] == value:
            class_counts[y] +=1      #class_counts是一个字典如：{'0'：20, '1':50, '2':30}表示：某feature_index对应的特征，\
            # 在类别为0的花中出现了20次,在类别为1的花中出现50次，在类别为2的花中出现了30次。
    sorted_class_counts = sorted(class_counts.items(), key = itemgetter, reverse = True)
    most_frequent_class = sorted_class_counts[0][0]   #得到排序后，某特征出现最多次数对应的那个类别。

#下面计算错误率
