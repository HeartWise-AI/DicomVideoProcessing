# README.md

## Overview

This repository contains Python code to convert DICOM (Digital Imaging and Communications in Medicine) files to AVI (Audio Video Interleave) files. This can be particularly useful when working with medical imaging data, as DICOM is a standard format for such data, while AVI is a more general-purpose video file format that can be easier to work with for certain applications.

## Features

The code scans through a given folder, including subfolders, to find and convert DICOM files.
It provides a function for masking and cropping movies.
Converts DICOM headers into a dictionary for easier data extraction.
Writes the converted video files to a specified destination folder.

## Dependencies

See requirements.txt

- cv2
- numpy
- pandas
- pydicom
- tqdm
- skimage
- gdcm
- pylibjpeg pylibjpeg-libjpeg

\*\*\* FOR GDCM \*\*\* :
`For GDCM, please refer to the official installation guide or consider using a pre-built docker image like pydicom/pydicom.`

## Python Scripts

1. `dicom_dataset_to_dict(dicom_header, fileToProcess)` : This function takes in a DICOM file header and converts it to a dictionary, skipping specific tags that have been marked for discard.

1. `extract_avi_and_metadata(path, data_type="ANGIO", destinationFolder="dicom_avi_extracted/", dataFolder="data/")` : This function is designed to extract AVI files and related metadata from DICOM files. It reads a CSV file containing paths to the DICOM files, converts each file to an AVI video file, and extracts metadata. The results are saved in the designated output directories.

1. `main(args=None)` : The main function that orchestrates the execution of the above functions. It utilizes argument parsing to accept command-line arguments and controls the flow of the script execution.

## Usage

This script is executed from the command line as follows:

```
python script.py --input_file input.csv --destinationFolder output_videos --dataFolder output_data --data_type ANGIO
```

```
import os

# Define the paths to the DICOM files and the destination folder
dicom_folder = "../CathAI/data/DeepCORO/CATHAI_Extracted_Concatenated/"
destination_folder = "data3/"

# Iterate through the DICOM files and convert them to AVI
for file_name in os.listdir(dicom_folder):
    file_to_process = os.path.join(dicom_folder, file_name)
    makeVideo(file_to_process, destination_folder)
```

Where:

- `--input_file` : The input CSV file containing the DICOM paths.

- `--destinationFolder` : The destination folder where the AVI files will be stored.

- `--dataFolder` : The destination folder where the data (metadata) will be stored.

- `--data_type` : The type of the data being processed. Default is "ANGIO".

## Disclaimer

These scripts have been created for use with DICOM medical imaging files. They should be used responsibly and ethically. Always ensure patient privacy and follow relevant laws and guidelines when handling medical data.

## License

This project is licensed under the terms of the MIT license.
