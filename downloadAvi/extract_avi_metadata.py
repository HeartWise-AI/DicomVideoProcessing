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
    dicom_dict = {}
    dicom_dict["file"] = fileToProcess
    repr(dicom_header)
    for dicom_value in dicom_header.values():
        try:
            if dicom_value.tag == (0x7FE0, 0x0010):
                # discard pixel data
                continue
            if dicom_value.tag == (0x7FDF, 0x1080):
                # discard [B-mode 1D Post Processing Curve]
                continue
            if dicom_value.tag == (0x7FDF, 0x1081):
                # discard [B-mode Delta (ECRI) Map Diagonal]
                continue
            if dicom_value.tag == (0x7FDF, 0x1085):
                # discard [Acoustic Frame Timestamp]
                continue
            if dicom_value.tag == (0x7FDF, 0x108D):
                # discard [ECG Data Value]
                continue
            if dicom_value.tag == (0x7FDF, 0x10F1):
                # discard [Trigger Mask.]
                continue
            if dicom_value.tag == (0x7FDF, 0xFE00):
                # discard [Data Padding]
                continue
            if dicom_value.tag == (0x7FDF, 0x1089):
                # discard [ECG Reference Timestamp]
                continue
            if dicom_value.tag == (0x7FDF, 0x1086):
                # discard [R-Wave Timestamp]
                continue
            if dicom_value.tag == (0x200D, 0x3000):
                # discard Private Data Tag
                continue
            if dicom_value.tag == (0x200D, 0x300F):
                # discard Private Data Tag
                continue
            if dicom_value.tag == (0x200D, 0x5100):
                # discard Private Data Tag
                continue
            #         if dicom_value.tag == (0x7fdf, 0x1086):
            #             # discard [R-Wave Timestamp]
            #             continue
            if type(dicom_value.value) == dicom.dataset.Dataset:
                dicom_dict[str(dicom_value.tag)] = dicom_dataset_to_dict(dicom_value.value)
            else:
                if dicom_value.tag == (0x18, 0x6011):
                    for i in range(int(str(dicom_value)[-2])):
                        for dicom_value_2 in dicom_value[i]:
                            v = _convert_value(dicom_value_2.value)
                            if int(str(dicom_value)[-2]) > 1:
                                if str(dicom_value_2.tag) in dicom_dict.keys():
                                    try:
                                        dicom_dict.setdefault(str(dicom_value_2.tag), []).append(
                                            v
                                        )
                                    except:
                                        (dicom_dict.setdefault(str(dicom_value_2.tag), []) + v)
                                else:
                                    dicom_dict[str(dicom_value_2.tag)] = [v]
                            else:
                                dicom_dict[str(dicom_value_2.tag)] = v
                elif dicom_value.tag == (0x32, 0x1064):
                    for dicom_value_4 in dicom_value[0]:
                        v = _convert_value(dicom_value_4.value)
                        dicom_dict[str(dicom_value_4.tag)] = v
                elif dicom_value.tag == (0x8, 0x2112):
                    for dicom_value_5 in dicom_value[0]:
                        v = _convert_value(dicom_value_5.value)
                        dicom_dict[str(dicom_value_5.tag)] = v
                elif dicom_value.tag == (0x10, 0x1002):
                    for dicom_value_6 in dicom_value[0]:
                        v = _convert_value(dicom_value_6.value)
                        dicom_dict[str(dicom_value_6.tag)] = v
                elif dicom_value.tag == (0x40, 0x0275):
                    for dicom_value_3 in dicom_value[0]:
                        v = _convert_value(dicom_value_3.value)
                        dicom_dict[str(dicom_value_3.tag)] = v
                elif dicom_value.tag == (0x0032, 0x1064):
                    for dicom_value_7 in dicom_value[0]:
                        v = _convert_value(dicom_value_7.value)
                        dicom_dict[str(dicom_value_7.tag)] = v
                elif dicom_value.tag == (0x0040, 0x0008):
                    for dicom_value_9 in dicom_value[0]:
                        v = _convert_value(dicom_value_9.value)
                        dicom_dict[str(dicom_value_9.tag)] = v
                elif dicom_value.tag == (0x0040, 0x0260):
                    for dicom_value_10 in dicom_value[0]:
                        v = _convert_value(dicom_value_10.value)
                        dicom_dict[str(dicom_value_10.tag)] = v
                else:
                    v = _convert_value(dicom_value.value)
                    # append if value is there, if not set key
                    if str(dicom_value.tag) in dicom_dict.keys():
                        try:
                            dicom_dict.setdefault(str(dicom_value.tag), []).append(v)
                        except:
                            dicom_dict.setdefault(str(dicom_value.tag), []) + v
                    else:
                        dicom_dict[str(dicom_value.tag)] = v
        except:
            continue
    return dicom_dict


def _sanitise_unicode(s):
    return s.replace("\u0000", "").strip()


def _convert_value(v):
    t = type(v)
    if t in (list, int, float):
        cv = v
    elif t == str:
        cv = _sanitise_unicode(v)
    elif t == bytes:
        s = v.decode("ascii", "replace")
        cv = _sanitise_unicode(s)
    elif t == dicom.valuerep.DSfloat:
        cv = float(v)
    elif t == dicom.valuerep.IS:
        cv = int(v)
    else:
        cv = repr(v)
    return cv


# In[165]:


def makeVideo(fileToProcess, destinationFolder, datatype="ANGIO"):
    fileName_2 = fileToProcess.split("/")[-2]  # \\ if windows, / if on mac or sherlock
    fileName_1 = fileToProcess.split("/")[
        -1
    ]  # hex(abs(hash(fileToProcess.split('/')[-1]))).upper()
    fileName = fileName_2 + "_" + fileName_1
    dicom_dict = {}
    if not os.path.isdir(os.path.join(destinationFolder, fileName)):
        dataset = dicom.dcmread(fileToProcess, force=True)
        try:
            testarray = dataset.pixel_array
            testarray = np.stack((testarray,) * 3, axis=-1)

            fps = 30

            if datatype == "ANGIO":
                fps = 15
                try:
                    fps = dataset[(0x08, 0x2144)].value
                except:
                    try:
                        fps = int(1000 / dataset[(0x18, 0x1063)].value)
                    except:
                        try:
                            fps = dataset[(0x18, 0x40)].value
                        except:
                            try:
                                fps = dataset[(0x7FDF, 0x1074)].value
                            except:
                                print("couldn't find frame rate, default to 15")
            else:
                try:
                    fps = dataset[(0x18, 0x1063)].value
                except:
                    try:
                        fps = dataset[(0x18, 0x40)].value
                    except:
                        try:
                            fps = dataset[(0x7FDF, 0x1074)].value
                        except:
                            print("couldn't find frame rate, default to 30")

            fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")

            ## If destinationFolder doesn't exist, create it
            if not os.path.isdir(destinationFolder):
                print("Creating ", destinationFolder)
                os.makedirs(destinationFolder)


            video_filename = os.path.join(destinationFolder, fileName + ".avi")
            # Requires for TTE:
            # finaloutput = dicom.pixel_data_handlers.convert_color_space(testarray,'YBR_FULL', 'RGB')
            # finaloutput = mask_and_crop(testarray)
            out = cv2.VideoWriter(video_filename, fourcc, fps, testarray.shape[1:3])
            try:
                for i in testarray:
                    out.write(i)
                out.release()
                dicom_dict = dicom_dataset_to_dict(dataset, fileToProcess)
                dicom_dict["video_path"] = video_filename
                return dicom_dict
            except:
                print("error", fileToProcess)
        except:
            print("Error in pixel data", fileToProcess)

    else:
        print(fileName, "hasAlreadyBeenProcessed")
    return 0


# In[156]:


def process_metadata(metadata, data_type):
    if data_type == "ANGIO":
        metadata = metadata[
            [
                "(0008, 0070)",
                "(0010, 0040)",
                "(0008, 2144)",
                "(0028, 0008)",
                "(0008, 0020)",
                "(0008, 0030)",
                "(0008, 0031)",
                "(0010, 0030)",
                "(0028, 0004)",
                "(0010, 0020)",
                "(0008, 0018)",
                "(0020, 000d)",
                "(0020, 000e)",
                "file",
                "video_path",
            ]
        ]

        metadata.columns = [
            "brand",
            "sex",
            "FPS",
            "NumberOfFrames",
            "date",
            "study_time",
            "series_time",
            "birthdate",
            "color_format",
            "mrn",
            "StudyID",
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "dicom_path",
            "FileName",
        ]

    elif data_type == "TTE":
        metadata = metadata[
            [
                "(0010, 0040)",
                "(0018, 1063)",
                "(0018, 0040)",
                "(7fdf, 1074)",
                "(0028, 0008)",
                "(0018, 602c)",
                "(0018, 602e)",
                "(0018, 1088)",
                "(0008, 0070)",
                "(0008, 0020)",
                "(0008, 0030)",
                "(0008, 1090)",
                "(0008, 1030)",
                "(0008, 1060)",
                "(0010, 0030)",
                "(0010, 1030)",
                "(0010, 1020)",
                "(0010, 21b0)",
                "(0010, 4000)",
                "(0028, 0004)",
                "(0010, 0020)",
                "(0008, 0018)",
                "(0020, 000d)",
                "(0020, 000e)",
                "file",
                "video_path",
            ]
        ]

        metadata.columns = [
            "sex",
            "FPS",
            "fps_2",
            "fps_3",
            "NumberOfFrames",
            "physical_delta_x",
            "physical_delta_y",
            "hr_bpm",
            "brand",
            "date",
            "time",
            "model",
            "study_type",
            "physician_reader",
            "birthdate",
            "patient_weight_kg",
            "patient_height_m",
            "patient_history",
            "patient_comments",
            "color_format",
            "mrn",
            "StudyID",
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "dicom_path",
            "FileName",
        ]
        metadata["FPS"] = metadata["FPS"].fillna(metadata["fps_3"])
    try:
        metadata["FPS"] = metadata["FPS"].fillna(metadata["fps_2"])
        metadata["FPS"] = metadata["FPS"].fillna(15.0)
    except:
        metadata["FPS"] = metadata["FPS"].fillna(15.0)

    try:
        metadata = metadata.drop(columns=["fps_2", "fps_3"])
    except:
        try:
            metadata = metadata.drop(columns=["fps_2"])
        except:
            print("No fps_2 column")
    metadata["StudyInstanceUID"] = metadata["StudyInstanceUID"].str.replace("'", "")
    return metadata


# In[177]:


def extract_avi_and_metadata(
    path,
    data_type="ANGIO",
    destinationFolder="dicom_avi_extracted/",
    dataFolder="data/",
):
    df = pd.read_csv(path)
    try:
        df["path"] = df["DICOMPath"]
    except:
        print("DICOMPath doesn't exist")

    count = 0
    final_list = []
    for index, row in tqdm(df.iterrows()):
        if not os.path.exists(destinationFolder):
            os.makedirs(destinationFolder)
            print("Making output directory as it doesn't exist", destinationFolder)
        count += 1
        VideoPath = os.path.join(row["path"])
        print(count, row["path"])
        if not os.path.exists(
            os.path.join(destinationFolder, row["path"][row["path"].rindex("/") + 1 :] + ".avi")
        ):
            dicom_metadata = makeVideo(VideoPath, destinationFolder)
            if isinstance(dicom_metadata, dict):
                final_list.append(dicom_metadata)
        else:
            print("Already did this file", VideoPath)
    dicom_df_final = pd.DataFrame(final_list)
    dicom_df_final = process_metadata(dicom_df_final, data_type)
    dicom_df_final["Split"] = "inference"
    fileName_1 = path.split("/")[-1]
    final_path = dataFolder + fileName_1 + "_metadata_extracted.csv"
    final_path_mu = dataFolder + fileName_1 + "_metadata_extracted_mu.csv"

    dicom_df_final.to_csv(final_path)
    reader = csv.reader(open(final_path), delimiter=",")
    writer = csv.writer(open(final_path_mu, "w"), delimiter="µ")
    writer.writerows(reader)
    return dicom_df_final


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(description="Predictions.")

    parser.add_argument("--input_file")
    parser.add_argument("--destinationFolder")
    parser.add_argument("--dataFolder")
    parser.add_argument("--data_type")

    parser = parser.parse_args(args)

    df = pd.read_csv(parser.input_file)
    print(df)
    extract_avi_and_metadata(
        parser.input_file,
        data_type=parser.data_type,
        destinationFolder=parser.destinationFolder,
        dataFolder=parser.dataFolder,
    )
    print("Done")


if __name__ == "__main__":
    main()
