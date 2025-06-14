import os
import sys
import argparse

from src import (
    ocr_extraction,
    data_cleaner,
    data_loader,
    download_dataset,
    check
)

def main():
    parser = argparse.ArgumentParser(description="Run full OCR and data processing pipeline.")
    parser.add_argument("--mode", type=str, choices=["download", "ocr", "clean", "load", "check", "all", "No"], default="all", help="Chooae a mode to continue")
    parser.add_argument("--input_path", type=str, help="Path to the input data (images or raw files).")
    parser.add_argument("--output_path", type=str, help="Path to save processed data.")
    #parser.add_argument("--dataset_url", type=str, help="Optional URL to download dataset if in 'download' mode.")
    args = parser.parse_args()

    if args.mode == "download":
        if not args.dataset_url:
            print("Error: --dataset_url is required for 'download' mode.")
            sys.exit(1)
        #download_dataset.run(args.dataset_url, args.output_path)
        download_dataset.run()

    if args.mode in ["ocr", "all"]:
        ocr_extraction.run()

    if args.mode in ["clean", "all"]:
        data_cleaner.run()

    if args.mode in ["load", "all"]:
        data_loader.run()

    if args.mode in ["check", "all"]:
        check.run()

    if args.mode == "No":
        print("exiting...")


if __name__ == "__main__":
    main()
