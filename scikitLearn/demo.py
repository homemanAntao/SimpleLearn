import mglearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 鸢尾花模型
iris_dataset = load_iris()
# print(iris_dataset)
print(iris_dataset['DESCR'][:193] + "\n......")
print(iris_dataset['target_names'])
print(iris_dataset['feature_names'])
# 样本数量 150 个样本 4个特征：花萼长度 花萼宽度 花瓣长度 花瓣宽度
print(iris_dataset['data'].shape)

print(iris_dataset['data'][:5])

# 被测量的品种
# print(type(iris_dataset['target']))
print(iris_dataset['target'].shape)

# 打乱数据集并进行拆分：（训练集不能 作为测试集使用(无法测试模型的泛化能力)， 比例：75% 训练集 25% 数据集）
# random_state 固定随机系数
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
print("X_train shape:{}".format(X_train.shape))
print("X_train shape:{}".format(y_train.shape))
print("X_train shape:{}".format(X_test.shape))
print("X_train shape:{}".format(y_test.shape))

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# 按照y_train着色，创建散点矩阵图

# 1、frame，pandas dataframe对象
# 2、alpha， 图像透明度，一般取(0,1]
# 3、figsize，以英寸为单位的图像大小，一般以元组 (width, height) 形式设置
# 4、ax，可选一般为none
# 5、diagonal，必须且只能在{‘hist', ‘kde'}中选择1个，'hist'表示直方图(Histogram plot),
# 'kde'表示核密度估计(Kernel Density Estimation)；该参数是scatter_matrix函数的关键参数
# 6、marker，Matplotlib可用的标记类型，如'.'，','，'o'等
# 7、density_kwds，(other plotting keyword arguments，可选)，与kde相关的字典参数
# 8、hist_kwds，与hist相关的字典参数
# 9、range_padding，(float, 可选)，图像在x轴、y轴原点附近的留白(padding)，该值越大，留白距离越大，图像远离坐标原点
# 10、kwds，与scatter_matrix函数本身相关的字典参数
# 11、c，颜色

grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20, },
                                 s=60,
                                 alpha=.8, cmap=mglearn.cm3)
plt.show()
