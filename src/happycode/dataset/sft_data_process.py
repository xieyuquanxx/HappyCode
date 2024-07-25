import json
import re

from tqdm import tqdm


with open("/data/Users/xyq/developer/happy_code/data/action_dpo/v1/mc_dataset_v1_img4_3.json", "rb") as f:
    datas = json.load(f)

system_prompt = "Based on current task, historical observations and actions, predict the four actions that masked as <action>.\n"


def process_prompt(data_string):
    # 定义正则表达式模式，匹配 "Current goal: " 后面跟随的任意字符，直到换行符
    pattern = r"Current goal: (.*?)\n"
    # 使用正则表达式查找匹配的内容
    match = re.search(pattern, data_string)
    if match:
        # 提取匹配的组（即 {task} 部分）
        task = match.group(1)
        return "Current task: " + task + "\n"
    else:
        return None


def process_img_dir(img_dir):
    img_lists = []
    for i in range(3):
        img_list = []
        img_list.append(img_dir[i])
        img_list.append(img_dir[i + 1])
        img_lists.append(img_list)
    return img_lists


def process_actions(data_string):
    # 分割字符串，以'<image_placeholder>'为分隔符，获取每两个之间的内容
    segments = re.split(r"(<image_placeholder>)", data_string)

    action_outputs = []
    action_inputs = []

    # 获取每两个<image_placeholder>之间的内容
    for i in range(1, len(segments) - 1, 2):
        segment = segments[i + 1]

        # 匹配所有的<a>{action}</a>序列
        actions = re.findall(r"(<a>.*?</a>)", segment)

        if len(actions) == 10:
            # 提取第4到第7个action
            action_output = "".join(actions[3:7])
            action_outputs.append(action_output)

            # 构建action_input，替换第4到第7个action为<action>
            actions[3:7] = ["<a><action></a>"] * 4
            action_input = "".join(actions)
            action_inputs.append(segments[i] + action_input + segments[i + 2])

    return action_outputs, action_inputs


def process_data(data):
    items = []
    action_outputs, action_inputs = process_actions(data["conversations"][1]["content"])
    img_lists = process_img_dir(data["conversations"][1]["images"])
    for i in range(3):
        item = {
            "conversations": [
                {
                    "role": "User",
                    "content": process_prompt(data["conversations"][0]["content"])
                    + system_prompt
                    + action_inputs[i],
                    "images": img_lists[i],
                },
                {"role": "Assistant", "content": action_outputs[i]},
            ]
        }
        items.append(item)
    return items


new_datas = []
for item in tqdm(datas):
    items = process_data(item)
    new_datas.extend(items)

with open("mc_dataset_sft_v1.json", "w", encoding="utf-8") as f:
    json.dump(new_datas, f, ensure_ascii=False)
