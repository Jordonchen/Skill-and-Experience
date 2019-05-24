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
    """
    计算某条规则的错误率，OneR算法会把具有某个特征的某个value的个体分到次数最多的类别中。错误率就是具有该value的个体在 \
    其他类别中出现的次数。其实就是计算class_counts字典中，除了most_frequent_class之外的个数。
     """
    error = sum([class_count for class_value, class_count in class_counts.items() if class_value != most_frequent_class])#如，这边输出{30，20]的sum，表示类别2 \
                                                                                                                          #和类别3对应的总数量
    return most_frequent_class, error
"""
下面可以用上面的函数去计算每一个特征值（0或者1）在每个类别中的错误率
"""
def train(X, y_value, feature_index):
    n_samples, n_features = X.shape
    assert 0 <= feature_index < n_features
    values = set(X[:,feature_index])  # 这边的作用：就是以数组的形式返回feature_index所指的列 \
                                    #   然后用set函数将数组转化为集合，找出几种不同的取值
    predictors = {}  #创建字典作为预测器，其中，键为特征值，值为类别。
    errors = []   #创建列表作为错误
    for current_value in values:
        most_frequent_class, error = train_feature_value(X, y_value, feature_index, current_value)
        predictors[current_value] = most_frequent_class
        errors.append(error)
    total_error = sum(errors)
    return predictors, total_error


"""
下面是进行测试的环节，之前已经把数据集划分成了X_train和X_test。使用训练集，计算所有特征的目标类别\
调用train_feature_value()训练预测器，计算错误率。对预测器进行排序，找到最佳的特征值，创建分类model。
"""
all_predictors = {variable: train(X_train, y_train, variable) for variable in range(X_train.shape[1])}#这边的variable就是train()里的feature_index \
# #同时，shape[1]表示读取列长度。如：此处输出{({'0':1,'1':0},error1),({'0':2,'1':1},error2),({'0':0,'1':1],error3),({'0':2,'1':0},error4)}
errors = {variable: error for variable, (mapping, error) in all_predictors.items()}#  如：{(0,error1),(1,error2),(2,error3),(3,error4)}
# Now choose the best and save that as "model"
# Sort by error
best_variable, best_error = sorted(errors.items(), key=itemgetter(1))[0]  #排序，得到的事元组形式
print("The best model is based on variable {0} and has error {1:.2f}".format(best_variable, best_error))

# Choose the bset model
model = {'variable': best_variable,
         'predictor': all_predictors[best_variable][0]}
print(model)     #如：输出：{'variable':2,'predictor':{0:0,1:2}}


#下面用模型对测试集的每条数据进行预测，得出结果
def predict(X_test, model):
    variable = model['variable']
    predictor = model['predictor']
    y_predicted = np.array([predictor[int(sample[variable])] for sample in X_test])
    return y_predicted

y_predicted = predict(X_test, model)
print(y_predicted)
print(y_test)
# 比较预测结果与实际的测试集类别，得出正确率。
accuracy = np.mean(y_predicted == y_test) * 100
print("The test accuracy is {:.1f}%".format(accuracy))

"""
正确率得到为65.8%，，对于OneR算法，这个正确率已经不错了，我们发现预测的结果类别中，只涉及到类别0和类别2，没有类别1 \
这就是使用一条规则可能带来的弊端。
"""
