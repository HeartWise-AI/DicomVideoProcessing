#!/usr/bin/env python

# In[162]:


# Notebook which iterates through a folder, including subfolders,
# and convert DICOM files to AVI files
import csv
import json
import multiprocessing
import os
import subprocess
import tempfile

import cv2

# function for masking and cropping echo movies
import numpy as np
import pandas as pd
import pydicom
from pydicom.uid import UID, generate_uid
from tqdm import tqdm

# sys.stdout = open(1, "w")
# python StudyInstanceUIDvi_metadata.py --input_file='../CathAI/data/DeepCORO/CATHAI_Extracted_Concatenated/DeepCORO_df_angle_object_dicom_2020_concat.csv' --destinationFolder='data3/' --dataFolder='data/' --data_type='ANGIO'


DICOM_DICT = {
    "ANGIO": {
        "(0008, 0070)": "brand",
        "(0010, 0040)": "sex",
        "(0008, 2144)": "FPS",
        "(0028, 0008)": "NumberOfFrames",
        "(0008, 0020)": "date",
        "(0008, 0030)": "study_time",
        "(0008, 0031)": "series_time",
        "(0010, 0030)": "birthdate",
        "(0028, 0004)": "color_format",
        "(0010, 0020)": "mrn",
        "(0008, 0018)": "StudyID",
        "(0020, 000d)": "StudyInstanceUID",
        "(0020, 000e)": "SeriesInstanceUID",
        "file": "dicom_path",
        "video_path": "FileName",
        "uint16_video": "uint16_video",
        "(0018, 1510)": "primary_angle",
        "(0018, 1511)": "secondary_angle",
        "(0028, 0010)": "width",
        "(0028, 0011)": "height",
        "(0028, 0030)": "pixel_spacing",
        "(0018, 1110)": "distance_source_to_detector",
        "(0018, 1111)": "distance_source_to_patient",
        "(0018, 1114)": "estimated_radiographic_magnification_factor",
        "(0018, 1134)": "table_motion",
        "(0018, 1155)": "radiation_setting",
        "(0018, 1164)": "image_pixel_spacing",
    },
    "TTE": {
        "(0010, 0040)": "sex",
        "(0018, 1063)": "FPS",
        "(0018, 0040)": "fps_2",
        "(7fdf, 1074)": "fps_3",
        "(0028, 0008)": "NumberOfFrames",
        "(0018, 602c)": "physical_delta_x",
        "(0018, 602e)": "physical_delta_y",
        "(0018, 1088)": "hr_bpm",
        "(0008, 0070)": "brand",
        "(0008, 0020)": "date",
        "(0008, 0030)": "time",
        "(0008, 1090)": "model",
        "(0008, 1030)": "study_type",
        "(0008, 1060)": "physician_reader",
        "(0010, 0030)": "birthdate",
        "(0010, 1030)": "patient_weight_kg",
        "(0010, 1020)": "patient_height_m",
        "(0010, 21b0)": "patient_history",
        "(0010, 4000)": "patient_comments",
        "(0028, 0004)": "color_format",
        "(0010, 0020)": "mrn",
        "(0008, 0018)": "StudyID",
        "(0020, 000d)": "StudyInstanceUID",
        "(0020, 000e)": "SeriesInstanceUID",
        "file": "dicom_path",
        "video_path": "FileName",
    },
}


def mask_and_crop(movie):
    from skimage import morphology

    sum_channel_mov = np.sum(movie, axis=3)
    diff_mov = np.diff(sum_channel_mov, axis=0)
    mask = np.sum(diff_mov.astype(bool), axis=0) > 10

    # erosion, followed by dilation to remove ecg traces touching cone
    selem = morphology.selem.disk(5)
    eroded = morphology.erosion(mask, selem)
    dilated = morphology.dilation(eroded, selem)

    # make mask 3-channel for more vectorized multiplication
    mask_3channel = np.zeros([dilated.shape[0], dilated.shape[1], 3])
    mask_3channel[:, :, 0] = dilated
    mask_3channel[:, :, 1] = dilated
    mask_3channel[:, :, 2] = dilated
    mask_3channel = mask_3channel.astype(bool)

    # get size of cropped movie
    x_locations = np.max(dilated, axis=0)
    y_locations = np.max(dilated, axis=1)
    left = np.where(x_locations)[0][0]
    right = np.where(x_locations)[0][-1]
    top = np.where(y_locations)[0][0]
    bottom = np.where(y_locations)[0][-1]
    h = bottom - top
    w = right - left

    # padding length for frame in x and y in case crop is beyond image boundaries
    pad = int(max([h, w]) / 2)
    x_center = right - int(w / 2) + pad
    y_center = bottom - int(h / 2) + pad

    # height and width of new frames
    size = int(max([h, w]) / 2) * 2
    crop_left = int(x_center - (size / 2))
    crop_right = int(x_center + (size / 2))
    crop_top = int(y_center - (size / 2))
    crop_bottom = int(y_center + (size / 2))

    # multiply each frame by mask, pad, and center crop
    masked_movie = movie
    out_movie = np.zeros([movie.shape[0], size, size, movie.shape[3]], dtype="uint8")
    for frame in range(movie.shape[0]):
        masked_frame = movie[frame, :, :] * mask_3channel
        padded_frame = np.pad(
            masked_frame,
            ((pad, pad), (pad, pad), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        out_movie[frame, :, :, :] = padded_frame[crop_top:crop_bottom, crop_left:crop_right, :]
    return out_movie


def process_metadata(metadata, data_type):
    tag_map = DICOM_DICT[data_type]

    # Convert metadata to DataFrame if it's not already
    if not isinstance(metadata, pd.DataFrame):
        metadata = pd.DataFrame([metadata])

    # Create a new DataFrame with all original columns
    processed_metadata = metadata.copy()

    # Rename columns based on tag_map
    rename_dict = {}
    for tag, col_name in tag_map.items():
        if tag in metadata.columns:
            rename_dict[tag] = col_name
        elif tag == "(0008, 2144)" and "CineRate" in metadata.columns:
            rename_dict["CineRate"] = col_name
        elif tag == "(0010, 0040)" and "PatientSex" in metadata.columns:
            rename_dict["PatientSex"] = col_name
        elif tag == "(0008, 0020)" and "StudyDate" in metadata.columns:
            rename_dict["StudyDate"] = col_name
        elif tag == "(0018, 1510)" and "PositionerPrimaryAngle" in metadata.columns:
            rename_dict["PositionerPrimaryAngle"] = col_name
        elif tag == "(0018, 1511)" and "PositionerSecondaryAngle" in metadata.columns:
            rename_dict["PositionerSecondaryAngle"] = col_name
        elif tag == "file" and "file" in metadata.columns:
            rename_dict["file"] = col_name
        elif tag == "video_path" and "video_path" in metadata.columns:
            rename_dict["video_path"] = col_name

    processed_metadata.rename(columns=rename_dict, inplace=True)

    # Handle FPS
    if "FPS" in processed_metadata.columns and processed_metadata["FPS"].isna().all():
        if "RecommendedDisplayFrameRate" in processed_metadata.columns:
            processed_metadata["FPS"] = processed_metadata["RecommendedDisplayFrameRate"]
        else:
            processed_metadata["FPS"] = 1.0  # default value

    # Handle StudyInstanceUID
    if "StudyInstanceUID" in processed_metadata.columns:
        processed_metadata["StudyInstanceUID"] = (
            processed_metadata["StudyInstanceUID"].astype(str).str.replace("'", "")
        )
    print(processed_metadata.columns.tolist())
    # Drop FrameTimeVector column
    if "FrameTimeVector" in processed_metadata.columns:
        processed_metadata = processed_metadata.drop(columns=["FrameTimeVector"])

    # print(f"Processed metadata shape: {processed_metadata.shape}")
    # print(f"Processed metadata columns: {processed_metadata.columns.tolist()}")

    return processed_metadata


def convert_to_serializable(obj):
    if isinstance(obj, (pydicom.multival.MultiValue, pydicom.valuerep.PersonName)):
        return str(obj)
    elif isinstance(obj, pydicom.valuerep.DSfloat):
        return float(obj)
    elif isinstance(obj, pydicom.valuerep.IS):
        return int(obj)
    elif isinstance(obj, pydicom.uid.UID):
        return str(obj)
    elif isinstance(obj, bytes):
        return obj.decode("utf-8", "ignore")
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def extract_h264_video_from_dicom(
    dicom_path, output_path, crf=23, preset="medium", data_type="ANGIO"
):
    dicom_data = pydicom.dcmread(dicom_path)
    pixel_array = dicom_data.pixel_array

    # Determine fps
    frame_rate = 15  # default fps
    frame_rate_tags = [(0x08, 0x2144), (0x18, 0x1063), (0x18, 0x40), (0x7FDF, 0x1074)]
    for tag in frame_rate_tags:
        try:
            frame_rate = float(dicom_data[tag].value)
            # print(f"Frame rate: {frame_rate}")
            break
        except (KeyError, AttributeError):
            print(f"Frame rate tag {tag} not found in DICOM file {dicom_path}")
            continue
    if data_type != "ANGIO" and frame_rate == 15:
        frame_rate = 30
    # print(f"Frame rate: {frame_rate}")

    if pixel_array.dtype != np.uint8:
        pixel_array = (
            (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
        ).astype(np.uint8)

    if len(pixel_array.shape) == 2:
        pixel_array = np.expand_dims(pixel_array, axis=0)

    is_color = pixel_array.shape[-1] == 3

    with tempfile.TemporaryDirectory() as temp_dir:
        for i, frame in enumerate(pixel_array):
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            if is_color:
                frame = frame[..., ::-1]  # Convert BGR to RGB if color
            cv2.imwrite(frame_path, frame)

        ffmpeg_command = [
            "ffmpeg",
            "-framerate",
            str(frame_rate),
            "-i",
            os.path.join(temp_dir, "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-y",
            output_path,
        ]

        try:
            subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg command failed: {e.stderr.strip()}")
            raise

    # Extract DICOM metadata
    dicom_dict = {
        elem.keyword: elem.value for elem in dicom_data.iterall() if elem.keyword != "PixelData"
    }
    dicom_dict["video_path"] = output_path

    return output_path, dicom_dict


if __name__ == "__main__":
    main()


def extract_h264_and_metadata(
    path,
    data_type="ANGIO",
    destinationFolder="dicom_h264_extracted/",
    dataFolder="data/",
    dicom_path_column="path",
    subdirectory=None,
    num_processes=None,
):
    from tqdm import tqdm

    try:
        df = pd.read_csv(path, sep="α", engine="python")
    except pd.errors.EmptyDataError:
        try:
            df = pd.read_csv(path, sep=",")
        except pd.errors.EmptyDataError:
            df = pd.read_csv(path, sep="μ")
    if not os.path.exists(dataFolder):
        os.makedirs(dataFolder)
        print("Making output directory as it doesn't exist", dataFolder)
    fileName_1 = path.split("/")[-1]
    final_path = os.path.join(dataFolder, fileName_1 + "_metadata_extracted.csv")
    final_path_alpha = os.path.join(dataFolder, fileName_1 + "_metadata_extracted_alpha.csv")
    if os.path.exists(final_path):
        raise FileExistsError(
            f"The file {final_path} already exists and cannot be created again."
        )

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            process_row,
            [
                (row, destinationFolder, subdirectory, dicom_path_column, data_type)
                for _, row in tqdm(df.iterrows())
            ],
        )
    final_list = [json.loads(result) for result in results if result is not None]

    if not final_list:
        print("No valid results were returned from processing.")
        return None

    dicom_df_final = pd.DataFrame(final_list)
    print("DataFrame created with shape:", dicom_df_final.shape)
    print("DataFrame columns:", dicom_df_final.columns.tolist())

    dicom_df_final = process_metadata(dicom_df_final, data_type)
    print("DataFrame after processing metadata:", dicom_df_final.shape)
    print("Final DataFrame columns:", dicom_df_final.columns.tolist())

    dicom_df_final.to_csv(final_path, index=False)
    print(f"Metadata saved to: {final_path}")

    # Write alpha-delimited file
    with open(final_path) as csvfile, open(final_path_alpha, "w", newline="") as alphafile:
        reader = csv.reader(csvfile)
        writer = csv.writer(alphafile, delimiter="α")
        for row in reader:
            writer.writerow(row)

    print(f"Alpha-delimited metadata saved to: {final_path_alpha}")

    return dicom_df_final


def process_row(row, destinationFolder, subdirectory, dicom_path_column, data_type="ANGIO"):
    if subdirectory:
        if subdirectory in row:
            subfolder = str(row[subdirectory]) + "/"
            destinationPath = os.path.join(destinationFolder, subfolder)
        else:
            raise ValueError(
                f"Subdirectory '{subdirectory}' is defined but not found in the row."
            )
    else:
        destinationPath = destinationFolder

    try:
        os.makedirs(destinationPath, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {destinationPath}: {str(e)}")

    dicom_path = os.path.join(row[dicom_path_column])
    output_filename = os.path.basename(dicom_path).replace(".dcm", ".mp4")
    output_path = os.path.join(destinationPath, output_filename)

    # Extract H.264 video and get metadata
    _, metadata = extract_h264_video_from_dicom(
        dicom_path, output_path, crf=23, preset="medium", data_type=data_type
    )

    # Convert metadata to serializable format

    serializable_metadata = {k: convert_to_serializable(v) for k, v in metadata.items()}

    # Add file and video_path to metadata if not already present
    if "file" not in serializable_metadata:
        serializable_metadata["file"] = dicom_path
    if "video_path" not in serializable_metadata:
        serializable_metadata["video_path"] = output_path

    # print(f"Processed DICOM: {dicom_path}")
    # print(f"Metadata: {serializable_metadata}")

    return json.dumps(serializable_metadata)  # Return serialized metadata


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract H.264 videos and metadata from DICOM files."
    )

    parser.add_argument("--input_file", required=True, help="Path to the input CSV file")
    parser.add_argument(
        "--dicom_path_column", required=True, help="Column name for DICOM file paths"
    )
    parser.add_argument(
        "--destinationFolder", required=True, help="Destination folder for extracted videos"
    )
    parser.add_argument("--subdirectory", help="Column name for subdirectory information")
    parser.add_argument(
        "--dataFolder", required=True, help="Folder for output metadata CSV files"
    )
    parser.add_argument("--data_type", default="ANGIO", help="Type of data (e.g., 'ANGIO')")
    parser.add_argument(
        "--num_processes",
        type=int,
        help="Number of processes to use (default: number of CPU cores)",
    )

    args = parser.parse_args(args)

    extract_h264_and_metadata(
        args.input_file,
        data_type=args.data_type,
        dicom_path_column=args.dicom_path_column,
        destinationFolder=args.destinationFolder,
        subdirectory=args.subdirectory,
        dataFolder=args.dataFolder,
        num_processes=args.num_processes,
    )
    print("Done")


if __name__ == "__main__":
    main()

# %%
