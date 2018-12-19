# -*- coding: utf-8 -*-

from C45 import *
import treePlotter
from logger import get_logger

logger = get_logger(path='log\\iris.log') # 日志存放文件

labels_dict = {
    'Sepal_Length': 'float64',
    'Sepal_Width': 'float64',
    'Petal_Length': 'float64',
    'Petal_Width': 'float64',
    'Class': 'object'
}
dataset_path = 'data/iris.txt'

logger.info('Data Loading ...')
data = pd.read_csv(dataset_path, names=list(labels_dict.keys()), dtype=labels_dict)  # 加载数据，设置数据类型，列名

Tree = C45() # 构建树
Tree.Load_data(data, test_size=0.5)  # 加载数据

# 训练数据集
logger.info('Training ...')
Tree.Train()
tree = Tree.tree
test_data = Tree.test_data.copy()

attribute_labels = data.columns.tolist()[:-1] # 属性列表
test_data_num = len(test_data) # 测试数据集大小
train_data_num = len(Tree.train_data) # 训练数据集大小
data_num = len(data) # 训练数据集和测试数据集总大小

# 预测数据
fun = lambda x:decision_tree_predict(one_test_data=list(x)[:], attribute_labels=attribute_labels, decision_tree=tree)
test_data['Class_pre'] = test_data.apply(fun, axis=1)
accuracy = sum(test_data['Class_pre']==test_data['Class']) / test_data_num # 准确率

logger.info(f'''
Data_set: {data_num}
Train_set: {train_data_num}
Test_set: {test_data_num}
Accuracy: {round(accuracy*100,2)}%
''')

logger.info(tree)
treePlotter.createPlot(tree)  # 画决策树
