def read_file(file_path: str):
    if file_path.endswith(".parquet"):
        from pandas import read_parquet

        data = read_parquet(file_path)
    elif file_path.endswith(".json"):
        import json

        data = json.load(open(file_path, "r"))
    return data
