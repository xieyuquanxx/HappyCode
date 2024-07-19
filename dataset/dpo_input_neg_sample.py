import json
import os
import random
from datetime import datetime

from tqdm import tqdm


current_date = datetime.now().strftime("%Y%m%d")


random.seed(42)

dataset_path = "/data/Users/xyq/developer/happy_code/data/action_dpo/v1"
dpo_file = f"{dataset_path}/mc_dataset_v1_img4_3.json"

img_dir = f"{dataset_path}/mc_dataset_v1"
all_tasks = os.listdir(img_dir)
task_dirs = {task: os.listdir(os.path.join(img_dir, task)) for task in all_tasks}


with open(dpo_file) as f:
    dpo_data = json.load(f)

print(len(dpo_data))  # 238873

for data in dpo_data:
    chosen = data["conversations"][1]
    images = chosen["images"]
    # /data/Users/xyq/developer/happy_code/dataset/mc_dataset_v1/craft_sticks_2522/craft_sticks_2522_frame_0.jpg
    task = "_".join(images[0].split("/")[-2].split("_")[:-1])
    filter_tasks = [t for t in all_tasks if task not in t]

    seleted_task = random.choice(filter_tasks)
    seleted_task_imgs = list(filter(lambda x: x.endswith(".jpg"), task_dirs[seleted_task]))

    images = random.choices(seleted_task_imgs, k=4)
    images = [os.path.join(dataset_path, img_dir, seleted_task, img) for img in images]
    input_neg = {
        "role": "User",
        "type": "rejected",
        "content": chosen["content"],
        "images": images,
    }
    data["conversations"].insert(2, input_neg)

with open(f"{dataset_path}/{current_date}_mc_dataset_v1_img4_input_1neg_output_3neg.json", "w") as fo:
    json.dump(dpo_data, fo)
