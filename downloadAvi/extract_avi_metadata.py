#!/usr/bin/env python

# In[162]:


# Notebook which iterates through a folder, including subfolders,
# and convert DICOM files to AVI files

import csv
import os
import os.path
import sys

import cv2
import numpy as np
import pandas as pd
import pydicom as dicom
from pydicom.uid import UID, generate_uid
from tqdm import tqdm

sys.stdout = open(1, "w")
# python StudyInstanceUIDvi_metadata.py --input_file='../CathAI/data/DeepCORO/CATHAI_Extracted_Concatenated/DeepCORO_df_angle_object_dicom_2020_concat.csv' --destinationFolder='data3/' --dataFolder='data/' --data_type='ANGIO'


# function for masking and cropping echo movies
import numpy as np
from skimage import morphology


def mask_and_crop(movie):
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


# In[164]:
def dicom_dataset_to_dict(dicom_header, fileToProcess):
    dicom_dict = {"file": fileToProcess}
    repr(dicom_header)

    discard_tags = [
        (0x7FE0, 0x0010),  # Pixel data
        (0x7FDF, 0x1080),  # B-mode 1D Post Processing Curve
        (0x7FDF, 0x1081),  # B-mode Delta (ECRI) Map Diagonal
        (0x7FDF, 0x1085),  # Acoustic Frame Timestamp
        (0x7FDF, 0x108D),  # ECG Data Value
        (0x7FDF, 0x10F1),  # Trigger Mask
        (0x7FDF, 0xFE00),  # Data Padding
        (0x7FDF, 0x1089),  # ECG Reference Timestamp
        (0x7FDF, 0x1086),  # R-Wave Timestamp
        (0x200D, 0x3000),  # Private Data Tag
        (0x200D, 0x300F),  # Private Data Tag
        (0x200D, 0x5100),  # Private Data Tag
    ]

    for dicom_value in dicom_header.values():
        try:
            if dicom_value.tag in discard_tags:
                continue

            if type(dicom_value.value) == dicom.dataset.Dataset:
                dicom_dict[str(dicom_value.tag)] = dicom_dataset_to_dict(
                    dicom_value.value, fileToProcess
                )
            else:
                v = _convert_value(dicom_value.value)
                if str(dicom_value.tag) in dicom_dict:
                    try:
                        dicom_dict.setdefault(str(dicom_value.tag), []).append(v)
                    except:
                        dicom_dict.setdefault(str(dicom_value.tag), []) + v
                else:
                    dicom_dict[str(dicom_value.tag)] = v
        except Exception as e:
            print(f"Error processing tag {dicom_value.tag}: {str(e)}")
            continue
    return dicom_dict


def _sanitise_unicode(s):
    return s.replace("\u0000", "").strip()


def _convert_value(v):
    t = type(v)
    if t in (list, int, float):
        return v
    elif t == str:
        return _sanitise_unicode(v)
    elif t == bytes:
        s = v.decode("ascii", "replace")
        return _sanitise_unicode(s)
    elif t == dicom.valuerep.DSfloat:
        return float(v)
    elif t == dicom.valuerep.IS:
        return int(v)
    else:
        return repr(v)


def normalize_16bit_to_8bit(array):
    """Normalize a 16-bit array to an 8-bit array."""
    array_min, array_max = array.min(), array.max()
    normalized_array = (array - array_min) / (array_max - array_min) * 255
    return normalized_array.astype(np.uint8)


def makeVideo(fileToProcess, destinationFolder, datatype="ANGIO"):
    fileName_2 = fileToProcess.split("/")[-2]  # \\ if windows, / if on mac or sherlock
    fileName_1 = fileToProcess.split("/")[
        -1
    ]  # hex(abs(hash(fileToProcess.split('/')[-1]))).upper()
    fileName = fileName_2 + "_" + fileName_1
    dicom_dict = {}
    uint16_value = False
    if not os.path.isdir(os.path.join(destinationFolder, fileName)):
        dataset = dicom.dcmread(fileToProcess, force=True)
        try:
            testarray = dataset.pixel_array

            # Check if the pixel data is in 16-bit format and normalize if necessary
            if testarray.dtype == np.uint16:
                testarray = normalize_16bit_to_8bit(testarray)
                uint16_value = True

            testarray = np.stack((testarray,) * 3, axis=-1)  # Convert to 3-channel

            # Determine fps
            fps = 15  # default fps
            frame_rate_tags = [(0x08, 0x2144), (0x18, 0x1063), (0x18, 0x40), (0x7FDF, 0x1074)]
            for tag in frame_rate_tags:
                try:
                    fps = dataset[tag].value
                    break
                except KeyError:
                    continue
            if datatype != "ANGIO" and fps == 15:
                fps = 30

            fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")

            # Create destination folder if it doesn't exist
            if not os.path.isdir(destinationFolder):
                print("Creating ", destinationFolder)
                os.makedirs(destinationFolder)
            video_filename = os.path.join(destinationFolder, fileName + ".avi")

            if not os.path.exists(video_filename):
                out = cv2.VideoWriter(video_filename, fourcc, fps, testarray.shape[1:3])

                # Write video
                try:
                    for i in testarray:
                        out.write(i)
                    out.release()
                    dicom_dict = dicom_dataset_to_dict(dataset, fileToProcess)
                    dicom_dict["video_path"] = video_filename
                    dicom_dict["uint16_video"] = uint16_value
                    return dicom_dict
                except Exception as e:
                    print(
                        f"Error while writing video for file: {fileToProcess}. Error details: {str(e)}"
                    )
            else:
                # print(f"Error: Video filename {video_filename} already exists.")
                dicom_dict = dicom_dataset_to_dict(dataset, fileToProcess)
                dicom_dict["video_path"] = video_filename
                dicom_dict["uint16_video"] = uint16_value
                return dicom_dict
        except Exception as e:
            print(f"Error in pixel data for file: {fileToProcess}. Error details: {str(e)}")
    else:
        print(fileName, "hasAlreadyBeenProcessed")

    return 0


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


def process_metadata(metadata, data_type):
    tag_map = DICOM_DICT[data_type]

    # Make sure all expected columns exist in the DataFrame
    for tag, col_name in tag_map.items():
        if tag not in metadata.columns:
            metadata[tag] = np.nan

    # Select only the columns of interest
    metadata = metadata[list(tag_map.keys())]

    # Rename the columns
    metadata.columns = list(tag_map.values())

    # Fill in the default FPS if not provided
    fps_col_names = ["FPS", "fps_2", "fps_3"]
    fps_col_names = [name for name in fps_col_names if name in metadata.columns]
    if fps_col_names:
        metadata.loc[:, "FPS"] = metadata[fps_col_names].bfill(axis=1).iloc[:, 0]
        metadata = metadata.drop(columns=fps_col_names[1:])
    else:
        metadata.loc[:, "FPS"] = 1.0  # default value if no FPS information is present
    metadata["StudyInstanceUID"] = metadata["StudyInstanceUID"].astype(str).str.replace("'", "")

    return metadata


# Helper function to convert DICOM dataset to dictionary
def dicom_dataset_to_dict(dicom_header, fileToProcess):
    dicom_dict = {"file": fileToProcess}
    for element in dicom_header:
        if element.VR != "SQ" and element.tag != (
            0x7FE0,
            0x0010,
        ):  # skip sequences and pixel data
            dicom_dict[element.keyword] = str(element.value)
    return dicom_dict


def extract_mp4_with_labels_and_metadata(
    df, path_column, label_column=None, second_label_column=None, destinationFolder=None
):
    """
    Saves .mp4 files with labels from label_column and second_label_column displayed on the video,
    and also saves metadata.

    Args:
        df: DataFrame containing the video file paths and labels.
        path_column: Column name containing the file paths.
        label_column: Column name for the primary label (optional).
        second_label_column: Column name for the secondary label (optional).
        destinationFolder: Folder where the output videos will be saved.
    """
    metadata_list = []
    for index, row in tqdm(df.iterrows()):
        dicom_path = row[path_column]
        try:
            dataset = dicom.dcmread(dicom_path, force=True)
            frames = dataset.pixel_array
        except Exception as e:
            print(f"Failed to read DICOM file {dicom_path}: {str(e)}")
            continue

        if frames.dtype == np.uint16:
            frames = (frames / frames.max()) * 255.0  # Normalize to 8-bit
            frames = frames.astype(np.uint8)

        if len(frames.shape) == 3:  # Assuming multiple frames
            height, width = frames.shape[1], frames.shape[2]
        else:
            height, width = frames.shape

        # Determine fps
        fps = 15  # default fps
        frame_rate_tags = [(0x08, 0x2144), (0x18, 0x1063), (0x18, 0x40), (0x7FDF, 0x1074)]
        for tag in frame_rate_tags:
            try:
                fps = dataset[tag].value
                break
            except KeyError:
                continue

        # Define codec and create VideoWriter object using MP4V codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        if destinationFolder and not os.path.exists(destinationFolder):
            os.makedirs(destinationFolder)
        output_path = os.path.join(
            destinationFolder if destinationFolder else "",
            dicom_path.split("/")[-1].replace(".dcm", "_labeled.mp4"),
        )
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        dicom_dict = dicom_dataset_to_dict(dataset, dicom_path)
        dicom_dict["FileName"] = output_path
        metadata_list.append(dicom_dict)

        if len(frames.shape) == 3:  # Assuming multiple frames, process each frame
            for frame in frames:
                if len(frame.shape) == 2:  # Convert grayscale to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                # Draw labels on the frame
                label_text = row[label_column] if label_column in df.columns else ""
                second_label_text = (
                    str(row[second_label_column]) if second_label_column in df.columns else ""
                )
                cv2.putText(
                    frame,
                    str(label_text),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    second_label_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                # Write the frame with labels
                out.write(frame)

        # Release everything when job is finished
        out.release()
        # print(f"Labeled video saved as {output_path}")

    # Save all metadata at once
    if destinationFolder and not os.path.exists(destinationFolder):
        os.makedirs(destinationFolder)
    metadata_filename = os.path.join(
        destinationFolder if destinationFolder else "", "all_metadata.csv"
    )
    metadata_df = pd.DataFrame(metadata_list)

    metadata_df.to_csv(metadata_filename, index=False)

    print(f"All metadata saved as {metadata_filename}")

    return metadata_df


if __name__ == "__main__":
    main()


def extract_avi_and_metadata(
    path,
    data_type="ANGIO",
    destinationFolder="dicom_avi_extracted/",
    dataFolder="data/",
    dicom_path_column="path",
    subdirectory=None,  # Adding the subdirectory parameter
):
    df = pd.read_csv(path)
    if not os.path.exists(dataFolder):
        os.makedirs(dataFolder)
        print("Making output directory as it doesn't exist", dataFolder)
    fileName_1 = path.split("/")[-1]
    final_path = os.path.join(dataFolder, fileName_1 + "_metadata_extracted.csv")
    final_path_mu = os.path.join(dataFolder, fileName_1 + "_metadata_extracted_mu.csv")
    if os.path.exists(final_path):
        raise FileExistsError(
            f"The file {final_path} already exists and cannot be created again."
        )

    final_list = []
    for count, (index, row) in enumerate(
        tqdm(df.iterrows(), desc="Processing rows", total=len(df))
    ):
        # Check if subdirectory parameter is used and valid
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

        if not os.path.exists(destinationPath):
            os.makedirs(destinationPath)
            print("Making output directory as it doesn't exist", destinationPath)

        try:
            VideoPath = os.path.join(row[dicom_path_column])
            outputFileName = (
                row[dicom_path_column][row[dicom_path_column].rindex("/") + 1 :] + ".avi"
            )
            dicom_metadata = makeVideo(VideoPath, destinationPath)
            if isinstance(dicom_metadata, dict):
                final_list.append(dicom_metadata)
        except:
            print("VideoPath doesn't exist", row[dicom_path_column])

    dicom_df_final = pd.DataFrame(final_list)

    dicom_df_final = process_metadata(dicom_df_final, data_type)
    dicom_df_final["Split"] = "inference"

    dicom_df_final.to_csv(final_path)
    reader = csv.reader(open(final_path), delimiter=",")
    print(final_path)
    writer = csv.writer(open(final_path_mu, "w"), delimiter="µ")
    writer.writerows(reader)
    return dicom_df_final


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(description="Predictions.")

    parser.add_argument("--input_file")
    parser.add_argument("--dicom_path_column")
    parser.add_argument("--destinationFolder")
    parser.add_argument("--subdirectory")
    parser.add_argument("--dataFolder")
    parser.add_argument("--data_type")

    parser = parser.parse_args(args)

    df = pd.read_csv(parser.input_file)
    extract_avi_and_metadata(
        parser.input_file,
        data_type=parser.data_type,
        dicom_path_column=parser.dicom_path_column,
        destinationFolder=parser.destinationFolder,
        subdirectory=parser.subdirectory,
        dataFolder=parser.dataFolder,
    )
    print("Done")


if __name__ == "__main__":
    main()

# %%
