import argparse
import json
import os
import pickle
import random
import re
from datetime import datetime

import numpy as np


current_date = datetime.now().strftime("%Y%m%d")


with open("dict_action.pkl", "rb") as f1:
    dic = pickle.load(f1)

with open("dict_action.pkl", "rb") as f1:
    new_dic = pickle.load(f1)


unused_action_key = ["chat", "pickItem", "swapHands"]
for uk in unused_action_key:
    del new_dic[uk]

# del new_dic["chat"]
# del new_dic["pickItem"]
# del new_dic["swapHands"]
action_types = list(new_dic.values())


def remove_last_segment(input_string):
    last_underscore_index = input_string.rfind("_")
    if last_underscore_index != -1:
        return input_string[:last_underscore_index]
    return input_string.replace("_", " ")


def insert_placeholders(text):
    # 先在开头插入一个 <image_placeholder>
    result = "<image_placeholder>"

    # 使用正则表达式找到所有 <a>{action}</a> 片段
    actions = re.findall(r"<a>.*?</a>", text)

    # 计数器，用于确定何时插入 <image_placeholder>
    count = 0

    for action in actions:
        # 每遍历十个 <a>{action}</a> 插入一个 <image_placeholder>
        count += 1
        if count % 10 == 0 and count != 0:
            result += "<image_placeholder>"
        result += action

    # 将多余的部分（非 <a>{action}</a> 的部分）添加到结果中
    remaining_text = re.split(r"(<a>.*?</a>)", text)
    remaining_text = [part for part in remaining_text if not re.match(r"<a>.*?</a>", part)]
    result += "".join(remaining_text)

    return result


def process_pkl_file(file_path):
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data
    except (OSError, pickle.UnpicklingError, FileNotFoundError) as e:
        print(f"Error processing {file_path}: {e}")
        return None


def input_farmat_transfer(input_action_list):
    action_sequence = ""
    for actions in input_action_list:
        action = ""
        if actions and actions != []:
            for a in actions:
                if isinstance(a, dict):
                    a = a["camera"]
                    x = f"{a[0]:.2f}"
                    y = f"{a[1]:.2f}"
                    action += f"<x>{x}</x>"
                    action += f"<y>{y}</y>"
                else:
                    action += a
        action_sequence += f"<a>{action}</a>"
    return insert_placeholders(action_sequence)


def output_farmat_transfer(output_action_list):
    action_sequence = ""
    for actions in output_action_list:
        action = ""
        if actions and actions != []:
            for a in actions:
                if type(a) == dict:
                    a = a["camera"]
                    x = f"{a[0]:.2f}"
                    y = f"{a[1]:.2f}"
                    action += f"<x>{x}</x>"
                    action += f"<y>{y}</y>"
                else:
                    action += a
        action_sequence += f"<a>{action}</a>"
    return action_sequence


def get_image_dir(base_directory, subfolder_path, index, group_size=4, begin_null_action_number: int = 0):
    start_id = index * group_size
    if index == 0 and begin_null_action_number > 0:
        start_id += begin_null_action_number
    end_id = start_id + group_size
    image_group = []

    for img_id in range(start_id, end_id):
        image_name = f"{subfolder_path}_frame_{img_id}.jpg"
        image_path = os.path.join(base_directory, subfolder_path, image_name)
        if os.path.exists(image_path):
            image_group.append(image_path)
        else:
            print(f"Image {image_name} does not exist in {subfolder_path}")
            return []

    return image_group


def camera_mapping(camera_value):
    camera_float = camera_value.tolist()
    if isinstance(camera_float[0], list):
        camera_float = camera_float[0]
    return camera_float


def adjust_sequence_length(action_sequence, clip_number: int = 80):
    length = len(action_sequence)
    if length > clip_number * 10:
        return action_sequence[: clip_number * 10 + 1]
    else:
        adjusted_length = (length // clip_number) * clip_number + 1
        return action_sequence[:adjusted_length]


def action_mapping(actions):
    action_list = []
    for action in actions:
        non_zero_keys = [key for key, value in action.items() if np.any(value != 0) and key not in unused_action_key]

        act = [dic[key] for key in non_zero_keys if key != "camera"]
        if "camera" in non_zero_keys:
            camera_value = camera_mapping(action["camera"])
            act.append({"camera": camera_value})
        action_list.append(act)
    assert len(actions) == len(action_list), "len(data) should equal to len(lst)"
    return action_list


def create_conversation(task, input_action_list, chosen_action_list, rejected_action_list, images_dir):
    input_action_list = input_farmat_transfer(input_action_list)
    chosen_action_list = output_farmat_transfer(chosen_action_list)
    rejected_action_list_1 = output_farmat_transfer(rejected_action_list[0])
    rejected_action_list_2 = output_farmat_transfer(rejected_action_list[1])
    rejected_action_list_3 = output_farmat_transfer(rejected_action_list[2])
    return {
        "conversations": [
            {
                "role": "System",
                "content": "Current goal: "
                + task
                + "\nPredict the next five actions based on historical observations actions.",
            },
            {"role": "User", "type": "chosen", "content": input_action_list, "images": images_dir},
            {"role": "Assistant", "type": "chosen", "content": chosen_action_list},
            {"role": "Assistant", "type": "rejected", "content": rejected_action_list_1},
            {"role": "Assistant", "type": "rejected", "content": rejected_action_list_2},
            {"role": "Assistant", "type": "rejected", "content": rejected_action_list_3},
        ]
    }


def create_conversation_v2(task, input_action_list, chosen_action_list, rejected_action_list, images_dir):
    input_action_list = input_farmat_transfer(input_action_list).split("<image_placeholder>")[1:]
    chosen_action_list = output_farmat_transfer(chosen_action_list)
    rejected_action_list_1 = output_farmat_transfer(rejected_action_list[0])
    rejected_action_list_2 = output_farmat_transfer(rejected_action_list[1])
    rejected_action_list_3 = output_farmat_transfer(rejected_action_list[2])
    return {
        "conversations": [
            {
                "role": "User",
                "type": "chosen",
                "content": "Current observations <image_placeholder>. Current goal: "
                + task
                + "\nPredict the next ten actions based on historical observations and actions.",
                "images": [images_dir[-1]],
                "history": {"images": images_dir[:-1], "actions": input_action_list[:8]},
            },
            {"role": "Assistant", "type": "chosen", "content": chosen_action_list},
            {"role": "Assistant", "type": "rejected", "content": rejected_action_list_1},
            {"role": "Assistant", "type": "rejected", "content": rejected_action_list_2},
            {"role": "Assistant", "type": "rejected", "content": rejected_action_list_3},
        ]
    }


def create_negative_samples(action_sequence, action_types, clip_number: int = 40):
    action_sequence = adjust_sequence_length(action_sequence, clip_number)
    negative_samples = []
    original_group_samples = []
    input_group_samples = []
    # 去掉第一个子列表
    action_sequence = action_sequence[1:]

    # 每40/80个子列表为一组
    num_groups = len(action_sequence) // clip_number

    for group_idx in range(num_groups):
        group_start = group_idx * clip_number
        group_end = group_start + clip_number
        group = action_sequence[group_start:group_end]

        group_negative_samples = []

        # 保存原始组的最后十个action
        input_group_samples.append(group[:-10])

        original_group_samples.append(group[-10:])

        # 只对每组子列表的最后十个action取样为负样本
        for _ in range(3):  # 每组序列取三组对应的负样本
            group_negative_sample = []
            for idx in range(len(group) - 10, len(group)):
                # original_actions = group[idx]
                negative_action = []
                num_actions = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1], k=1)[0]  # 非等概率取1到3个action

                for _ in range(num_actions):
                    new_action = random.choice(action_types)
                    # 避免重复添加相同的action
                    while new_action in negative_action:
                        new_action = random.choice(action_types)
                    if new_action == "<camera>":
                        new_action = {"camera": [round(random.uniform(-10, 10), 3), round(random.uniform(-10, 10), 3)]}
                    negative_action.append(new_action)

                group_negative_sample.append(negative_action)

            group_negative_samples.append(group_negative_sample)

        negative_samples.append(group_negative_samples)

    return num_groups, input_group_samples, original_group_samples, negative_samples


def process_all_subfolders(base_directory, args):
    # from tqdm import tqdm

    action_data = []
    action_dataset = []
    for subfolder in os.listdir(base_directory):
        subfolder_path = os.path.join(base_directory, subfolder)
        if os.path.isdir(subfolder_path):
            pkl_file_path = os.path.join(subfolder_path, f"{subfolder}.pkl")
            if os.path.exists(pkl_file_path):
                datas = process_pkl_file(pkl_file_path)
                if datas is not None:
                    action_list = action_mapping(datas)

                    # 去除开头空的action，并记录个数
                    null_action_number = 0
                    for action in action_list:
                        if len(action) == 0:
                            null_action_number += 1
                        else:
                            break
                    action_list = action_list[null_action_number:]

                    num_groups, input_action_list, chosen_action_list, rejected_action_list = create_negative_samples(
                        action_list, action_types, clip_number=args.clip_number
                    )
                    for i in range(num_groups):
                        dic = {}
                        task = remove_last_segment(subfolder)
                        dic["data_id"] = subfolder
                        # dic['raw_actions']=action_list
                        dic["img_dir"] = subfolder_path
                        dic["clip_id"] = str(i)
                        dic["input_actions"] = input_action_list[i]
                        dic["choosen"] = chosen_action_list[i]  # clip_num, action_sequence
                        dic["rejected_1"] = rejected_action_list[i][0]
                        dic["rejected_2"] = rejected_action_list[i][1]
                        dic["rejected_3"] = rejected_action_list[i][2]
                        action_data.append(dic)
                        images_dir = get_image_dir(
                            base_directory,
                            subfolder,
                            i,
                            group_size=args.group_image_num,
                            begin_null_action_number=null_action_number,
                        )
                        if len(images_dir) < 9:
                            continue
                        action_dataset.append(
                            create_conversation_v2(
                                task,
                                input_action_list[i],
                                chosen_action_list[i],
                                rejected_action_list[i],
                                images_dir,
                            )
                        )
            else:
                print(f"No .pkl file found in {subfolder_path}")

    print(f"Dataset Len: {len(action_dataset)}")
    with open(
        f"{args.output_dir}/{current_date}_{args.output}".replace(".json", "_action.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(action_data, f, ensure_ascii=False)
    with open(f"{args.output_dir}/{current_date}_{args.output}", "w", encoding="utf-8") as f:
        json.dump(action_dataset, f, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--group_image_num", default=9, type=int)
    parser.add_argument("--output", type=str, default="mc_dataset_v2_img8.json")
    parser.add_argument("--output_dir", default="/data/Users/xyq/developer/happy_code/data/action_vlm")
    parser.add_argument("--clip_number", default=80, type=int)
    parser.add_argument(
        "--base_dir", default="/data/Users/xyq/developer/happy_code/data/mc_dataset/mc_dataset_v2", type=str
    )
    args = parser.parse_args()

    # base_directory = "/data/Users/xyq/developer/happy_code/data/action_dpo/v1/mc_dataset_v2"  # 主文件夹路径
    process_all_subfolders(args.base_dir, args)
