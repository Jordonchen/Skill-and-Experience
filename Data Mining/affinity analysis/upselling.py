# coding:utf-8
# Copyright:Learning Data Mining with Python  by  Robert Layton
"""
这是一个亲和性分析的例子，顾客在买多种商品时，有时会存在某些“规律”，\
商家可以了解一下顾客在买一件商品时，同时会买其他什么商品。如果数据足够，可以用亲和性分析来对数据进行分析，\
把顾客愿意同时购买的商品放一起，可以提高销售额。
"""
import numpy as np
from collections import defaultdict

dataset_filename = "affinity_dataset.txt"  #load the file
X = np.loadtxt(dataset_filename)
n_samples, n_features = X.shape  # 100 samples and 5 features
print("This dataset has {0} samples and {1} features".format(n_samples,n_features))

features = ["bread", "milk", "cheese", "apples", "bananas"]

num_apple_purchases = 0
for sample in  X:
    if sample[3] == 1:
        num_apple_purchases += 1
print("{0} people buy apples".format(num_apple_purchases))  #举例输出多少顾客买了苹果。

"""
我们定义一个有效规则例子：如果一个人买了苹果（规则前提），他也会买香蕉（规则结果）。
"""
rule_valid = 0
rule_invalid = 0
for sample in X:
    if sample[3] ==1:
        if sample[4] ==1:
            rule_valid +=1
        else:
            rule_invalid +=1
print("{0} cases is valid becase the person bought apples and bananas".format(rule_valid))
print("{0} cases is invaid because the person bought apples but not bananas".format(rule_invalid))
"""
现在计算支持度和置信度，支持度就是规则应验的次数，\
置信度就是规则应验次数/规则的前提（如果有人买了苹果）出现的次数
"""
support = rule_valid
confidence = rule_valid / num_apple_purchases
print("support is {0} and confidence is {1:.3f}".format(support,confidence))

"""计算所有的规则的support和confidence，需要用到字典存放结果,用到collections的defaultdict模块"""
valid_rules = defaultdict(int)
invalid_rules = defaultdict(int)
num_occurances = defaultdict(int)
for sample in X:
     for premise in range(n_features):
         if sample[premise] == 0:
             continue
         num_occurances[premise] += 1
         for conclusion in range(n_features):
             if premise == conclusion:
                 continue
             if sample[conclusion] == 1:
                 valid_rules[(premise,conclusion)] += 1
             else:
                 invalid_rules[(premise,conclusion)] += 1
support = valid_rules
confidence = defaultdict(float)
for premise,conclusion in valid_rules.keys():
    confidence[(premise,conclusion)] = valid_rules[(premise,conclusion)] / num_occurances[premise]

def print_rule(premise, conclusion, support, confidence, features):
    premise_name = features[premise]
    conclusion_name = features[conclusion]
    print("if a persion buys {0} he will also buy {1}".format(premise_name,conclusion_name))
    print("support is {0}".format(support[(premise,conclusion)]))
    print("confidence is {0:.3f}".format(confidence[(premise,conclusion)]))

premise = 1
conclusion = 3
print_rule(premise, conclusion, support, confidence, features)


"""
我们对所有的规则的支持度进行一个排序，从大到小排序，找出支持度最大的那一条规则。\ 
字典排序使用itemgetter()类。
"""
# items()返回字典所有元素的列表，item getter（1）表示以字典元素的值作为排序依据，reverse = True表示降序
sorted_support = sorted(support.items(), key = itemgetter(1), reverse = True)
print(sorted_support)

for index in range(5):    #输出支持度的排序前五个。
    print("Rule #{0}".format(index + 1))
    premise, conclusion = sorted_support[index][0]
    print_rule(premise, conclusion, support, confidence, features)

sorted_confidence = sorted(confidence.items(), key = itemgetter(1), reverse = True)
print(sorted_confidence)
for index in range(5): #    输出置信度排序前五个
    print("Rule #{0}".format(index + 1))
    premise, conclusion = sorted_confidence[index][0]
    print_rule(premise, conclusion, support, confidence, features)





