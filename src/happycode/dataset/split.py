import argparse
import json
import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        help="Input file",
        default="/data/Users/xyq/developer/happy_code/data/action_dpo/v2/20240722_mc_dataset_v2_img8.json",
    )
    parser.add_argument("--split_ratio", type=float, nargs="+", default=[0.6, 0.3, 0.1])
    args = parser.parse_args()

    random.seed(42)

    with open(args.data_file) as f:
        data = json.load(f)

    print(f"{args.data_file} has {len(data)} samples.")

    lens = len(data)
    split_ratios = args.split_ratio

    chunk_sizes = [int(lens * ratio) for ratio in split_ratios]
    chunk_sizes[-1] = lens - sum(chunk_sizes[:-1])

    random.shuffle(data)

    chunks = []
    for size in chunk_sizes:
        chunk, data = data[:size], data[size:]
        chunks.append(chunk)

    for i, chunk in enumerate(chunks):
        with open(f"{args.data_file.replace(".json", "")}_{len(chunk)}.json", "w") as f:
            json.dump(chunk, f)

    print("done :)")
