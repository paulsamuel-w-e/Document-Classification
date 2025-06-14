from datasets import load_dataset
from pathlib import Path
import argparse
import os


def get_datasets(file_path: str, val_size: float, test_size: float):
    # Load your Parquet file
    dataset = load_dataset("parquet", data_files={"data" : file_path})

    train_val_test = dataset['data'].train_test_split(test_size=val_size + test_size, seed=42)
    val_test = train_val_test['test'].train_test_split(test_size=test_size / (val_size + test_size), seed=42)

    train_dataset = train_val_test['train']
    val_dataset = val_test['train']
    test_dataset = val_test['test']


    return train_dataset, val_dataset, test_dataset

def main():
    parser = argparse.ArgumentParser(description="Load datasets from Parquet file")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the Parquet file")
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--output_path", type=str, default="./Data/split", help="Path to save the split datasets")

    file_path = parser.parse_args().file_path
    try:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"The specified file does not exist: {file_path}")
        if Path(file_path).suffix == '.parquet':
            train_dataset, val_dataset, test_dataset = get_datasets(file_path, parser.parse_args().val_size, parser.parse_args().test_size)
            print(f"Train dataset size: {len(train_dataset)}")
            print(f"Validation dataset size: {len(val_dataset)}")
            print(f"Test dataset size: {len(test_dataset)}")

            if not Path(parser.parse_args().output_path).exists():
                os.makedirs(Path(parser.parse_args().output_path), exist_ok=True)
                print(f"Output directory created at: {parser.parse_args().output_path}")
            else:
                print(f"Output directory exists at: {parser.parse_args().output_path}")
            # Save datasets as Parquet files
            train_dataset.to_parquet(Path(parser.parse_args().output_path) / "train_dataset.parquet")
            val_dataset.to_parquet(Path(parser.parse_args().output_path) / "val_dataset.parquet")
            test_dataset.to_parquet(Path(parser.parse_args().output_path) / "test_dataset.parquet")
            print("Datasets saved as Parquet files.")
        else:
            raise ValueError("The provided file is not a Parquet file. Please provide a valid Parquet file.")
        return train_dataset, val_dataset, test_dataset

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    except ValueError as e:
        print(f"Error: {e}")
        return
    
main() if __name__ == "__main__" else None