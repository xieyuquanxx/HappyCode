import os
import glob
import pickle
import numpy as np
import json
from collections import OrderedDict


with open('dict_action.pkl', 'rb') as f1:
    dic=pickle.load(f1)

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
    negative_samples = []
    for actions in action_sequence:
        negative_action = []
        for action in actions:
            if isinstance(action, str):
                # 随机选择一个不同的 action 替换原来的 action
                new_action = random.choice(action_types)
                while new_action == action:
                    new_action = random.choice(action_types)
                negative_action.append(new_action)
            elif isinstance(action, dict) and 'camera' in action:
                # 对于 'camera' action，直接添加到负样本中
                negative_action.append(action)
        negative_samples.append(negative_action)
    return negative_samples

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
                    dic['data_id']=subfolder
                    dic['actions']=action_list
                    dic['img_dir']=subfolder_path
                    action_data.append(dic)
            else:
                print(f"No .pkl file found in {subfolder_path}")
    with open("test.json",'w',encoding='utf-8') as f:
        json.dump(action_data, f,ensure_ascii=False)





if __name__ == "__main__":
    base_directory = "mc_dataset_test"  # 主文件夹路径
    process_all_subfolders(base_directory)