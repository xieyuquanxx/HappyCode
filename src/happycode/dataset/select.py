import argparse
import json
import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", help="Input file")
    parser.add_argument("--output_file", help="Output file")
    parser.add_argument("--sample_number", type=int, help="Number of samples")
    args = parser.parse_args()

    random.seed(42)

    with open(args.data_file) as f:
        data = json.load(f)

    print(f"{args.data_file} has {len(data)} samples. Selecting {args.sample_number} samples...")

    data = random.choices(data, k=args.sample_number)

    with open(args.output_file, "w") as f:
        json.dump(data, f)

    print("done :)")
