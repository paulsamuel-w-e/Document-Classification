from datasets import load_dataset
from pathlib import Path
import argparse
import os

def get_datasets(file_path: str, val_size: float, test_size: float):
    dataset = load_dataset("parquet", data_files={"data": file_path})
    train_val_test = dataset['data'].train_test_split(test_size=val_size + test_size, seed=42)
    val_test = train_val_test['test'].train_test_split(test_size=test_size / (val_size + test_size), seed=42)
    return train_val_test['train'], val_test['train'], val_test['test']

def main():
    parser = argparse.ArgumentParser(description="Load and split Parquet dataset")
    parser.add_argument("--file_path", type=str, default ="./Data/extracts/documents.parquet",required=True, help="Path to the Parquet file")
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--output_path", type=str, default="./Data/split", help="Path to save split datasets")

    args = parser.parse_args()

    try:
        file = Path(args.file_path)
        if not file.exists():
            raise FileNotFoundError(f"File does not exist: {file}")
        if file.suffix != '.parquet':
            raise ValueError("Provided file is not a Parquet file.")

        train_ds, val_ds, test_ds = get_datasets(str(file), args.val_size, args.test_size)

        print(f"Train dataset size: {len(train_ds)}")
        print(f"Validation dataset size: {len(val_ds)}")
        print(f"Test dataset size: {len(test_ds)}")

        output_dir = Path(args.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_path = output_dir / "train_dataset.parquet"
        val_path = output_dir / "val_dataset.parquet"
        test_path = output_dir / "test_dataset.parquet"

        train_ds.to_parquet(train_path)
        val_ds.to_parquet(val_path)
        test_ds.to_parquet(test_path)

        print("Datasets saved as Parquet files.")
        return {"train": train_path, "val": val_path, "test": test_path}

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

main() if __name__ == "__main__" else None