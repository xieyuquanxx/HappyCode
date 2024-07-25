import json
import os
import random
from datetime import datetime


current_date = datetime.now().strftime("%Y%m%d")


random.seed(42)

dataset_path = "/data/Users/xyq/developer/happy_code/data/action_dpo/v2"
dpo_file = f"{dataset_path}/20240722_mc_dataset_v2_img8.json_18782.json"

img_dir = "/data/Users/xyq/developer/happy_code/data/action_dpo/mc_dataset_v2"
all_tasks = os.listdir(img_dir)
task_dirs = {task: os.listdir(os.path.join(img_dir, task)) for task in all_tasks}


with open(dpo_file) as f:
    dpo_data = json.load(f)

print(len(dpo_data))

for data in dpo_data:
    chosen = data["conversations"][0]
    images = chosen["images"]
    # /data/Users/xyq/developer/happy_code/dataset/mc_dataset_v1/craft_sticks_2522/craft_sticks_2522_frame_0.jpg
    task = "_".join(images[0].split("/")[-2].split("_")[:-1])
    filter_tasks = [t for t in all_tasks if task not in t]

    seleted_task = random.choice(filter_tasks)
    seleted_task_imgs = list(filter(lambda x: x.endswith(".jpg"), task_dirs[seleted_task]))

    images = random.choices(seleted_task_imgs, k=9)
    images = [os.path.join(dataset_path, img_dir, seleted_task, img) for img in images]
    input_neg = {
        "role": "User",
        "type": "rejected",
        "content": chosen["content"],
        "images": [images[-1]],
        "history": {"images": images[:-1], "actions": chosen["history"]["actions"]},
    }
    data["conversations"].insert(2, input_neg)

with open(f"{dataset_path}/{current_date}_mc_dataset_v2_img8_input_1neg_output_3neg.json", "w") as fo:
    json.dump(dpo_data, fo)


print("done :)")
print(dpo_data[0])
