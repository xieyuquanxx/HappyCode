
import pickle


# with open("/data/Users/xyq/developer/happy_code/dataset/mc_dataset_sft_v1.json", encoding="utf-8") as load_f:
#     datas = json.load(load_f)

# print(len(datas))

# print(datas[0])



# for data in datas:
#     if data["conversations"][1]["images"] == []:
#         print(data["conversations"][0])

# datas = [data for data in datas if data["conversations"][1]["images"] != []]
# print(len(datas))
# with open("mc_dataset_v1_img4_2.json", "w", encoding="utf-8") as f:
#     json.dump(datas, f, ensure_ascii=False)

# for i in range(len(datas)):
#    assert len(datas[i]['rejected'][0])==3,'not equal'
# print(len(datas))

# with open("/data/Users/xyq/developer/happy_code/dataset/mc_dataset_test/chop_a_tree_3/chop_a_tree_3.pkl",'rb') as f:
#    data=pickle.load(f)


# for action in data:
#    print(action['camera'])

with open("dict_action.pkl", "rb") as f1:
    dic = pickle.load(f1)


special_tokens_list=[]
for key,value in dic.items():
    special_tokens_list.append(value)


