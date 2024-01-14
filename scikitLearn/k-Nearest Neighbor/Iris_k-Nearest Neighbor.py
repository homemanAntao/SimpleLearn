# 所谓K近邻算法，即是给定一个训练数据集，对新的输入实例，在训练数据集中找到与该实例最邻近的K个实例（也就是上面所说的K个邻居），
# 这K个实例的多数属于某个类，就把该输入实例分类到这个类中。
# KNN 算法本身简单有效，它是一种 lazy-learning 算法
# K近邻算法使用的模型实际上对应于对特征空间的划分。K 值的选择，距离度量和分类决策规则是该算法的三个基本要素

# 1. K值的选取 一般选取一个数值 采取交叉取值的方法
#第一种，简单交叉验证
# 所谓的简单，是和其他交叉验证方法相对而言的。首先，我们随机的将样本数据分为两部分（比如： 50%的训练集，50%的验证集），然后用训练集来训练模型，
# 在验证集上验证模型及参数。接着，我们再把样本打乱，重新选择训练集和验证集，继续训练数据和检验模型。最后我们选择损失函数评估最优的模型和参数。　

# 特点：在分割训练集和验证集的时候，一般是随机的，这也正是容易出猫腻的地方。同一个模型和参数，分别在两次分割中，验证结果可能会出现很大的差距。所以，不太有说服力。

# 第二种，K折交叉验证（K-Folder Cross Validation）
# 和第一种方法不同，K折交叉验证会把样本数据随机的分成K份，每次随机的选择K-1份作为训练集，剩下的1份做验证集。当这一轮完成后，重新随机选择K-1份来训练数据。若干轮（小于K）之后，选择损失函数评估最优的模型和参数。

# 特点：K一般大于等于2，实际操作时一般从3开始取，只有在原始数据集合数据量小的时候才会尝试取 2。它可以有效的避免过拟合以及欠拟合状态的发生，最后得到的结果也比较具有说服力。

# 第三种，留一手交叉验证（Leave-one-out Cross Validation）
# 它是第二种情况的特例，此时K等于样本数N，这样对于N个样本，每次选择N-1个样本来训练数据，留一个样本来验证模型预测的好坏。此方法主要用于样本量非常少的情况，比如对于普通适中问题，N小于50时，我一般采用留一交叉验证。

# 特点：结果比较可靠。过程中没有随机因素会影响结果，确保了整个过程是可以被复


# ； 2.特征量归一化（例如买鞋问题 男性身高和尺码为特性向量 ）

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
# print("X_train shape:{}".format(X_train.shape))
# print("X_train shape:{}".format(y_train.shape))
# print("X_train shape:{}".format(X_test.shape))
# print("X_train shape:{}".format(y_test.shape))

knn = KNeighborsClassifier(1)
knn.fit(X_train, y_train)
print(knn)

X_new = np.array([[5, 2.9, 1, 0.2]])
print(X_new.shape)

prediction = knn.predict(X_new)
print("prediction :{}".format(prediction))
print("prediction name:{}".format(iris_dataset['target_names'][0]))

# 模型评估
y_prediction = knn.predict(X_test)
print(y_prediction)
#print("test set score: {:.2f}".format(np.mean(y_prediction == y_test)))
print("test set score: {:.2f}".format(knn.score(X_test == y_test)))
