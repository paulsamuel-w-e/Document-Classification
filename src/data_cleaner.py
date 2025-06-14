# ./src/data_cleaner.py

# Importing Necessary libraries
import gc
import regex as re
from pathlib import Path
import simplejson
import tqdm
import argparse
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Load the JSON file
def load_text_data(file_path: Path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return simplejson.load(f)


# Remove the Unwanted entries in the JSON File
def remove_text_data(json_list: list, index_list: list):
    for idx in sorted(index_list, reverse=True):
        if 0 <= idx < len(json_list):
            del json_list[idx]
    return json_list


# Write back the JSON File
def write_text_data(file_path: Path, data: list):
    with open(file_path, 'w', encoding='utf-8') as f:
        simplejson.dump(data, f, ensure_ascii=False, indent=4)

# Clean bounding boxes to rect
def polygon_to_rect(polygon):
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]

# Reorder keys in the dictionary to a desired order
def reorder_keys(d):
    desired_order = ["id", "label", "image_path", "full_text", "words", "bbox", "confidence"]
    return {k: d[k] for k in desired_order if k in d}

# Regex Pipeline
def perform_regex(text_path: Path):
    doc = load_text_data(text_path)

    cleaned_texts = []
    removed_indices = []


    for idx, text in enumerate(doc['full_text']):
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'[^\w\s!"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~]', '', text)
        text = re.sub(r'(?<!\w)[!"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~](?!\w)', '', text)
        text = text.strip()
        
        if text:
            cleaned_texts.append(text)
        else:
            removed_indices.append(idx)

    sorted_desc = sorted(removed_indices, reverse=True)

    cleaned_conf = remove_text_data(doc['confidence'], sorted_desc)
    cleaned_bbox = remove_text_data(doc['bbox'], sorted_desc)
    cleaned_texts = remove_text_data(doc['full_text'], sorted_desc)


    doc['full_text'] = " ".join(cleaned_texts).strip()
    doc['words'] = cleaned_texts
    doc['bbox'] = [polygon_to_rect(poly) for poly in cleaned_bbox]
    doc['confidence'] = cleaned_conf

    return reorder_keys(doc)


# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default = "./Data/extracts/ocr_texts", help = "Path to your ocr extracted texts directory")
    args = parser.parse_args()

    #making empty list for our list of jsons
    dataset = []

    input_path = Path(args.input_dir)
    loop = tqdm.tqdm(input_path.iterdir(), desc="Processing directories")

    for dir_path in loop:
        if dir_path.is_dir():
            for file_path in dir_path.iterdir():
                if not file_path.name.endswith('.json'):
                    print(f"skipping non-json file {file_path.parts[-1]}")
                    continue
                text_path = file_path
                # tqdm.tqdm.write(f"Cleaning: {file_path.name}")
                doc  = perform_regex(text_path)
                dataset.append(doc)

        gc.collect()

    write_text_data(Path("./Data/extracts/cleaned_dataset.json"), dataset)

    file = ".\Data\extracts\cleaned_dataset.json"
    data = load_text_data(file)
    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, ".\Data\extracts\documents.parquet")
    print("Data cleaning and conversion to parquet completed successfully.")
    gc.collect()