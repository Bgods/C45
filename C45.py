# -*- coding: utf-8 -*-

from math import log
from sklearn.model_selection import train_test_split
import pandas as pd
import copy


class C45():
    def __init__(self):
        self.tree = {}
        self.train_data = pd.DataFrame()
        self.test_data = pd.DataFrame()

    def Load_data(self, data, test_size=0.33):
        self.labels = list(data.columns)
        self.train_data, self.test_data = train_test_split(data,test_size=test_size)  # 使用train_test_split函数将源数据集分割成训练数据集和测试数据集


    def Train(self):
        labels = copy.deepcopy(self.labels)
        self.tree = self.CteateTree(self.train_data, labels)

    # 创建决策树：决策树主程序
    def CteateTree(self, dataSet, labels):

        categoryList = dataSet.iloc[:,-1] # 抽取源数据集中的类别标签列

        # 决策树终止条件1：如果所有的数据都属于同一个类别，则返回该类别
        if len(set(categoryList)) == 1:
            return categoryList.values[0]

        # 决策树终止条件2：如果数据没有属性值数据，则返回该其中出现最多的类别作为分类
        if len(dataSet.columns.values) == 1:
            return self.most_voted_attribute(label_list=list(categoryList))


        # 获取最佳属性列名(best_attribute)，和最佳分割点(best_split_point)
        best_attribute, best_split_point = self.attribute_selection_method(dataSet)

        if best_attribute == None:
            return self.most_voted_attribute(label_list=list(categoryList))

        labels.remove(best_attribute)  # 找到最佳划分属性后需要将其从属性名列表中删除

        decision_tree = {best_attribute: {}}

        # 如果best_split_point为空，说明此时最佳划分属性的类型为离散值
        if best_split_point == None:

            attribute_list = list(dataSet[best_attribute].unique())
            for attribute in attribute_list:  # 属性的各个值
                sub_labels = labels[:]
                sub_data = self.split_data_set(dataSet, best_attribute, attribute, continuous=False)
                decision_tree[best_attribute][attribute] = self.CteateTree(sub_data, sub_labels)

        # 如果best_split_point不为空，说明此时最佳划分属性的类型为连续型
        else:
            sub_labels = labels[:]

            sub_data = self.split_data_set(dataSet, best_attribute, best_split_point, continuous=True,part=0)
            decision_tree[best_attribute]["<=" + str(round(best_split_point,4))] = self.CteateTree(sub_data, sub_labels)

            sub_labels = labels[:]
            sub_data = self.split_data_set(dataSet, best_attribute, best_split_point, continuous=True, part=1)
            decision_tree[best_attribute][">" + str(round(best_split_point,4))] = self.CteateTree(sub_data, sub_labels)
        return decision_tree


    def attribute_selection_method(self, dataSet):

        '''
        选择属性列方法: 分别求各个属性的信息增益率，找出信息增益率最大的属性
        :param dataSet: 原数据集合
        :return: 信息增益率最大的属性列索引
        '''
        num_attributes = dataSet.columns.__len__()-1  # 属性的个数，减1是因为去掉了标签
        info_D = self.cate_info(dataSet)  # 香农熵

        max_grian_rate = 0.0  # 最大信息增益比
        #best_attribute_index = -1
        best_attribute = None
        best_split_point = None

        for i in range(num_attributes):

            info_A_D = 0.0  # 特征A对数据集D的信息增益
            split_info_D = 0.0  # 数据集D关于特征A的值的熵

            attribute_columns = dataSet.columns[i]

            # 判断第i属性列，是否为连续型数值
            dtype = dataSet.iloc[:, i].dtype
            if dtype=='float64' or dtype=='int64':
                continuous = True
            else:
                continuous = False

            """
            属性为连续值，先对该属性下的所有离散值进行排序
            然后每相邻的两个值之间的中点作为划分点计算信息增益比，对应最大增益比的划分点为最佳划分点
            由于可能多个连续值可能相同，所以通过set只保留其中一个值
            """

            if continuous == True:  # 连续型数值
                attribute_list = dataSet.iloc[:,i].unique() # 将第i列属性提取出来，并且去重
                attribute_list.sort() # 排序

                split_points = []

                for index in range(len(attribute_list) - 1):
                    # 求出各个划分点
                    split_points.append((float(attribute_list[index]) + float(attribute_list[index + 1])) / 2)

                for split_point in split_points:  # 对划分点进行遍历
                    info_A_D = 0.0
                    split_info_D = 0.0

                    for part in range(2):  # 最佳划分点将数据一分为二，因此循环2次即可得到两段数据

                        sub_data_set = self.split_data_set(dataSet, attribute_columns, split_point, True, part)
                        prob = len(sub_data_set) / float(len(dataSet))
                        info_A_D += prob * self.cate_info(sub_data_set)
                        split_info_D -= prob * log(prob, 2)

                    if split_info_D == 0:
                        split_info_D += 1

                    """
                    由于关于属性A的熵split_info_D可能为0，因此需要特殊处理
                    常用的做法是把求所有属性熵的平均，为了方便，此处直接加1
                    """
                    grian_rate = (info_D - info_A_D) / split_info_D  # 计算信息增益比
                    if grian_rate > max_grian_rate:
                        max_grian_rate = grian_rate
                        best_split_point = split_point
                        best_attribute = attribute_columns

            else:  # 划分属性为离散值
                attribute_list = list(dataSet.iloc[:, i].unique()) # 求属性列表

                for attribute in attribute_list:  # 对每个属性进行遍历，求数据集D关于特征A的值的熵
                    sub_data_set = self.split_data_set(dataSet, attribute_columns, attribute, False)  # 提取第i列属性值为attribute的数据集合
                    prob = len(sub_data_set) / float(len(dataSet))
                    info_A_D += prob * self.cate_info(sub_data_set)
                    split_info_D -= prob * log(prob, 2)

                if split_info_D == 0:
                    split_info_D += 1

                grian_rate = (info_D - info_A_D) / split_info_D  # 计算属性attribute的信息增益率

                if grian_rate > max_grian_rate:
                    max_grian_rate = grian_rate
                    best_attribute = attribute_columns
                    best_split_point = None  # 如果最佳属性是离散值，此处将分割点置为空留作判定

        return best_attribute, best_split_point


    def split_data_set(self, dataSet, best_attribute, value, continuous, part=0):
        if continuous == True:  # 划分的属性为连续值
            if part == 0:  # 求划分点左侧的数据集
                res_data_set = dataSet[dataSet[best_attribute] <= value]
                res_data_set = res_data_set.drop(best_attribute, axis=1)

            if part == 1:  # 求划分点右侧的数据集
                res_data_set = dataSet[dataSet[best_attribute] > value]
                res_data_set = res_data_set.drop(best_attribute, axis=1)

        else:  # 划分的属性为离散值
            res_data_set = dataSet[dataSet[best_attribute] == value]
            res_data_set = res_data_set.drop(best_attribute, axis=1)

        return res_data_set


    # 计算数据集的类别信息熵
    def cate_info(self, dataSet):
        if len(dataSet.columns.values) == 1:
            return 0.0
        else:
            class_name = dataSet.columns[-1] # 获取最后一列名(类别名)
            data_by = dataSet.groupby(by=class_name).count().iloc[:, -1] # 按照类别分组
            num_entries = dataSet.__len__() # 求数据集行数
            # 计算信息熵
            return sum(data_by.apply(lambda x: -(x / num_entries) * log(x / num_entries, 2)))


    # 返回列表label_list中，出现频率最高的类别
    def most_voted_attribute(self, label_list):
        labels = set(label_list)
        max_label = ''
        max_num = 0
        for label in labels:
            if label_list.count(label) > max_num:
                max_num = label_list.count(label)
                max_label = label
        return max_label


# 预测数据集
def decision_tree_predict(decision_tree, attribute_labels, one_test_data):
    first_key = list(decision_tree.keys())[0]
    second_dic = decision_tree[first_key]
    attribute_index = attribute_labels.index(first_key)
    res_label = None
    for key in second_dic.keys():  # 属性分连续值和离散值，连续值对应<=和>两种情况
        if key[0] == '<':
            value = float(key[2:])
            if float(one_test_data[attribute_index]) <= value:
                if type(second_dic[key]).__name__ == 'dict':
                    res_label = decision_tree_predict(second_dic[key], attribute_labels, one_test_data)
                else:
                    res_label = second_dic[key]

        elif key[0] == '>':
            # print(key[1:])
            value = float(key[1:])
            if float(one_test_data[attribute_index]) > value:
                if type(second_dic[key]).__name__ == 'dict':
                    res_label = decision_tree_predict(second_dic[key], attribute_labels, one_test_data)
                else:
                    res_label = second_dic[key]

        else:
            if one_test_data[attribute_index] == key:
                if type(second_dic[key]).__name__ == 'dict':
                    res_label = decision_tree_predict(second_dic[key], attribute_labels, one_test_data)
                else:
                    res_label = second_dic[key]
    return res_label



