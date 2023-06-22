# README.md

## Overview

This repository contains a collection of Python scripts to extract metadata and convert DICOM files to AVI. This process is very useful in the field of medical imaging where DICOM is a standard format. The scripts contain various functions for handling DICOM files including extracting data from DICOM headers and converting the data to AVI files.

## Python Scripts

1. `dicom_dataset_to_dict(dicom_header, fileToProcess)` : This function takes in a DICOM file header and converts it to a dictionary, skipping specific tags that have been marked for discard. 

2. `extract_avi_and_metadata(path, data_type="ANGIO", destinationFolder="dicom_avi_extracted/", dataFolder="data/")` : This function is designed to extract AVI files and related metadata from DICOM files. It reads a CSV file containing paths to the DICOM files, converts each file to an AVI video file, and extracts metadata. The results are saved in the designated output directories. 

3. `main(args=None)` : The main function that orchestrates the execution of the above functions. It utilizes argument parsing to accept command-line arguments and controls the flow of the script execution. 

## Usage 

This script is executed from the command line as follows:

```python
python script.py --input_file input.csv --destinationFolder output_videos --dataFolder output_data --data_type ANGIO
```

Where:

- `--input_file` : The input CSV file containing the DICOM paths.

- `--destinationFolder` : The destination folder where the AVI files will be stored.

- `--dataFolder` : The destination folder where the data (metadata) will be stored.

- `--data_type` : The type of the data being processed. Default is "ANGIO".

## Dependencies

- pandas
- tqdm
- os
- argparse
- csv

## Additional Notes

- The `makeVideo` and `process_metadata` functions are called within `extract_avi_and_metadata` but are not defined in this README. Please ensure these functions are defined and working as expected in your script.
- The function `dicom_dataset_to_dict` also uses an undefined function `_convert_value`. Ensure this function is defined in your script.
  
## Disclaimer

These scripts have been created for use with DICOM medical imaging files. They should be used responsibly and ethically. Always ensure patient privacy and follow relevant laws and guidelines when handling medical data.
