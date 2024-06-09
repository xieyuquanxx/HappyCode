from datasets import DatasetDict


def train_test_split(dataset, test_radio: float = 0.2):
    # 80% train, 20% validation
    split_dataset = dataset["train"].train_test_split(test_size=test_radio)
    # 50% validation, 50% test
    eval_dataset = split_dataset["test"].train_test_split(test_size=0.5)

    our_dataset = DatasetDict(
        {
            "train": split_dataset["train"],
            "validation": eval_dataset["train"],
            "test": eval_dataset["test"],
        }
    )
    return our_dataset
