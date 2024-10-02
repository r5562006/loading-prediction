# -*- coding: UTF-8 -*-
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator
import pickle

def calc_shannon_entropy(data_set):
    """計算給定數據集的經驗熵（香農熵）"""
    num_entries = len(data_set)  
    label_counts = {}  
    
    for feat_vec in data_set:  
        current_label = feat_vec[-1]  
        if current_label not in label_counts:  
            label_counts[current_label] = 0
        label_counts[current_label] += 1  
    
    shannon_entropy = 0.0  
    
    for key in label_counts:  
        prob = float(label_counts[key]) / num_entries  
        shannon_entropy -= prob * log(prob, 2)  
    
    return shannon_entropy  

def create_data_set():
    """創建測試數據集"""
    data_set = [[0, 0, 0, 0, 'no'],  
                [0, 0, 0, 1, 'no'],
                [0, 1, 0, 1, 'yes'],
                [0, 1, 1, 0, 'yes'],
                [0, 0, 0, 0, 'no'],
                [1, 0, 0, 0, 'no'],
                [1, 0, 0, 1, 'no'],
                [1, 1, 1, 1, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [2, 0, 1, 2, 'yes'],
                [2, 0, 1, 1, 'yes'],
                [2, 1, 0, 1, 'yes'],
                [2, 1, 0, 2, 'yes'],
                [2, 0, 0, 0, 'no']]
    labels = ['年齡', '有工作', '有自己的房子', '信貸情況']  
    return data_set, labels  

def split_data_set(data_set, axis, value):  
    """按照給定特徵劃分數據集"""
    ret_data_set = []  
    
    for feat_vec in data_set:  
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]  
            reduced_feat_vec.extend(feat_vec[axis + 1:])  
            ret_data_set.append(reduced_feat_vec)
    
    return ret_data_set  

def choose_best_feature_to_split(data_set):
    """選擇最優特徵"""
    num_features = len(data_set[0]) - 1  
    base_entropy = calc_shannon_entropy(data_set)  
    best_info_gain = 0.0  
    best_feature = -1  
    
    for i in range(num_features):  
        feat_list = [example[i] for example in data_set]
        unique_vals = set(feat_list)  
        new_entropy = 0.0  
        
        for value in unique_vals:  
            sub_data_set = split_data_set(data_set, i, value)  
            prob = len(sub_data_set) / float(len(data_set))  
            new_entropy += prob * calc_shannon_entropy(sub_data_set)  
        
        info_gain = base_entropy - new_entropy  
        
        if info_gain > best_info_gain:  
            best_info_gain = info_gain  
            best_feature = i  
    
    return best_feature  

def majority_count(class_list):
    """統計classList中出現此處最多的元素（類標籤）"""
    class_count = {}
    
    for vote in class_list:  
        if vote not in class_count: 
            class_count[vote] = 0  
        class_count[vote] += 1
    
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)  
    return sorted_class_count[0][0]  

def create_tree(data_set, labels, feat_labels):
    """創建決策樹"""
    class_list = [example[-1] for example in data_set]  
    
    if class_list.count(class_list[0]) == len(class_list):  
        return class_list[0]
    
    if len(data_set[0]) == 1 or len(labels) == 0:  
        return majority_count(class_list)
    
    best_feat = choose_best_feature_to_split(data_set)  
    best_feat_label = labels[best_feat]  
    feat_labels.append(best_feat_label)
    my_tree = {best_feat_label: {}}  
    del(labels[best_feat])  
    
    feat_values = [example[best_feat] for example in data_set]  
    unique_vals = set(feat_values)  
    
    for value in unique_vals:  
        sub_labels = labels[:]  
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels, feat_labels)
        
    return my_tree

def get_num_leafs(my_tree):
    """獲取決策樹葉子節點的數目"""
    num_leafs = 0  
    first_str = next(iter(my_tree))  
    second_dict = my_tree[first_str]  
    
    for key in second_dict.keys():
        if isinstance(second_dict[key], dict):  
            num_leafs += get_num_leafs(second_dict[key])
        else:   
            num_leafs += 1
            
    return num_leafs

def get_tree_depth(my_tree):
    """獲取決策樹的層數"""
    max_depth = 0  
    first_str = next(iter(my_tree))  
    second_dict = my_tree[first_str]  
    
    for key in second_dict.keys():
        if isinstance(second_dict[key],
