import os
import glob
import pickle
import numpy as np
import json
from collections import OrderedDict
import random

with open('dict_action.pkl', 'rb') as f1:
    dic=pickle.load(f1)


del dic['pickItem']
del dic['swapHands']
print(dic.values())
action_types = list(dic.values())


def process_pkl_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except (pickle.UnpicklingError, FileNotFoundError, IOError) as e:
        print(f"Error processing {file_path}: {e}")
        return None
    
def camera_mapping(camera_value):
    camera_float = camera_value.tolist()
    return camera_float

def adjust_sequence_length(action_sequence):
    length = len(action_sequence)
    if length > 400:
        return action_sequence[:401]
    else:
        adjusted_length = (length // 40) * 40 + 1
        return action_sequence[:adjusted_length]
    
def action_mapping(actions):
    action_list=[]
    for action in actions:
        non_zero_keys = [key for key, value in action.items() if np.any(value != 0)]
        
        act=[dic[key] for key in non_zero_keys if key != 'camera']
        if 'camera' in non_zero_keys:
            camera_value = action['camera']
            camera_value = camera_mapping(camera_value)
            act.append({'camera':camera_value})
        action_list.append(act)
    assert len(actions)==len(action_list),"len(data) should equal to len(lst)"
    return action_list

def create_negative_samples(action_sequence, action_types):
    action_sequence = adjust_sequence_length(action_sequence)
    negative_samples = []
    original_group_samples = []

    # 去掉第一个子列表
    action_sequence = action_sequence[1:]

    # 每40个子列表为一组
    num_groups = len(action_sequence) // 40

    for group_idx in range(num_groups):
        group_start = group_idx * 40
        group_end = group_start + 40
        group = action_sequence[group_start:group_end]

        group_negative_samples = []

        # 保存原始组的最后十个action
        original_group_sample = group[30:40]
        original_group_samples.append([original_group_sample])

        # 只对每组子列表的最后十个action取样为负样本
        for _ in range(3):  # 每组序列取三组对应的负样本
            group_negative_sample = []
            for idx in range(30, 40):
                original_actions = group[idx]
                negative_action = []
                num_actions = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1], k=1)[0]  # 非等概率取1到3个action

                for _ in range(num_actions):
                    new_action = random.choice(action_types)
                    # 避免重复添加相同的action
                    while new_action in negative_action:
                        new_action = random.choice(action_types)
                    negative_action.append(new_action)

                group_negative_sample.append(negative_action)
            
            group_negative_samples.append(group_negative_sample)
        
        negative_samples.append(group_negative_samples)

    return num_groups,original_group_samples,negative_samples


def process_all_subfolders(base_directory):
    action_data=[]
    for subfolder in os.listdir(base_directory):
        subfolder_path = os.path.join(base_directory, subfolder)
        if os.path.isdir(subfolder_path):
            pkl_file_path = os.path.join(subfolder_path, f"{subfolder}.pkl")
            if os.path.exists(pkl_file_path):
                datas = process_pkl_file(pkl_file_path)
                if datas is not None:
                    dic={}
                    action_list=action_mapping(datas)
                    num_groups,chosen_action_list,rejected_action_list=create_negative_samples(action_list, action_types)
                    dic['data_id']=subfolder
                    dic['raw_actions']=action_list
                    dic['img_dir']=subfolder_path
                    dic['clip_num']=num_groups
                    dic['choosen']=chosen_action_list
                    dic['rejected']=rejected_action_list
                    action_data.append(dic)
            else:
                print(f"No .pkl file found in {subfolder_path}")
    with open("test.json",'w',encoding='utf-8') as f:
        json.dump(action_data, f,ensure_ascii=False)





if __name__ == "__main__":
    base_directory = "mc_dataset_test"  # 主文件夹路径
    process_all_subfolders(base_directory)