#!/usr/bin/env python
"""
Notebook which iterates through a folder, including subfolders,
and converts DICOM (and NPZ) files to MP4 videos (or PNG images when single‐frame).
"""

import csv
import json
import multiprocessing
import os
import subprocess
import tempfile

import cv2
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_color_lut
from pydicom.uid import UID, generate_uid
from tqdm import tqdm

# Dictionary to map DICOM tags to simplified column names
DICOM_DICT = {
    "ANGIO": {
        "(0008, 0070)": "brand",
        "(0010, 0040)": "sex",
        "(0008, 2144)": "FPS",
        "(0028, 0008)": "NumberOfFrames",
        "(0008, 0020)": "date",
        "(0008, 0030)": "StudyTime",
        "(0008, 0031)": "SeriesTime",
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
        "video_path": "video_path",
    },
}


def convert_to_serializable(obj, max_length=5000):
    """
    Convert pydicom / numpy data types to standard Python types for JSON.
    We also skip or truncate large binary fields (e.g., waveforms).
    """
    import pydicom
    from pydicom.dataset import Dataset as DicomDataset
    from pydicom.sequence import Sequence as DicomSequence

    if isinstance(obj, (pydicom.multival.MultiValue, pydicom.valuerep.PersonName)):
        return str(obj)
    elif isinstance(obj, pydicom.valuerep.DSfloat):
        return float(obj)
    elif isinstance(obj, pydicom.valuerep.IS):
        return int(obj)
    elif isinstance(obj, pydicom.uid.UID):
        return str(obj)
    elif isinstance(obj, bytes):
        if len(obj) > max_length:
            return f"<binary data, length={len(obj)} truncated>"
        try:
            return obj.decode("utf-8", "ignore")[:max_length]
        except UnicodeDecodeError:
            return f"<binary data, length={len(obj)}>"
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        if obj.size > max_length:
            return f"<large ndarray, shape={obj.shape}, dtype={obj.dtype}>"
        else:
            return obj.tolist()
    elif isinstance(obj, DicomSequence):
        return [convert_to_serializable(ds, max_length) for ds in obj]
    elif isinstance(obj, DicomDataset):
        serial_dict = {}
        for elem in obj.iterall():
            tag_str = f"({elem.tag.group:04x}, {elem.tag.element:04x})"
            if elem.keyword in ["PixelData", "WaveformData", "CurveData"]:
                continue
            if tag_str in ["(5000, 3000)", "(7fe0, 0010)"]:
                continue
            serial_dict[elem.keyword] = convert_to_serializable(elem.value, max_length)
        return serial_dict
    elif isinstance(obj, list):
        return [convert_to_serializable(item, max_length) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v, max_length) for k, v in obj.items()}
    return obj


def mask_and_crop(movie):
    from skimage import morphology

    sum_channel_mov = np.sum(movie, axis=3)
    diff_mov = np.diff(sum_channel_mov, axis=0)
    mask = np.sum(diff_mov.astype(bool), axis=0) > 10

    selem = morphology.selem.disk(5)
    eroded = morphology.erosion(mask, selem)
    dilated = morphology.dilation(eroded, selem)

    mask_3channel = np.zeros([dilated.shape[0], dilated.shape[1], 3])
    mask_3channel[:, :, 0] = dilated
    mask_3channel[:, :, 1] = dilated
    mask_3channel[:, :, 2] = dilated
    mask_3channel = mask_3channel.astype(bool)

    x_locations = np.max(dilated, axis=0)
    y_locations = np.max(dilated, axis=1)
    left = np.where(x_locations)[0][0]
    right = np.where(x_locations)[0][-1]
    top = np.where(y_locations)[0][0]
    bottom = np.where(y_locations)[0][-1]
    h = bottom - top
    w = right - left

    pad = int(max([h, w]) / 2)
    x_center = right - int(w / 2) + pad
    y_center = bottom - int(h / 2) + pad

    size = int(max([h, w]) / 2) * 2
    crop_left = int(x_center - (size / 2))
    crop_right = int(x_center + (size / 2))
    crop_top = int(y_center - (size / 2))
    crop_bottom = int(y_center + (size / 2))

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

    if not isinstance(metadata, pd.DataFrame):
        metadata = pd.DataFrame([metadata])

    processed_metadata = metadata.copy()
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

    if "FPS" in processed_metadata.columns and processed_metadata["FPS"].isna().all():
        if "RecommendedDisplayFrameRate" in processed_metadata.columns:
            processed_metadata["FPS"] = processed_metadata["RecommendedDisplayFrameRate"]
        else:
            processed_metadata["FPS"] = 1.0

    if "StudyInstanceUID" in processed_metadata.columns:
        processed_metadata["StudyInstanceUID"] = (
            processed_metadata["StudyInstanceUID"].astype(str).str.replace("'", "")
        )
    if "FrameTimeVector" in processed_metadata.columns:
        processed_metadata = processed_metadata.drop(columns=["FrameTimeVector"])

    return processed_metadata


def extract_h264_video_from_dicom(
    dicom_path, output_path, crf=23, preset="medium", data_type="ANGIO", lossless=False
):
    """
    Read a DICOM file, process its pixel data according to its Photometric Interpretation,
    and then save as an MP4 video (if multi-frame) or PNG image (if single-frame).
    Returns the output file path and serializable metadata.
    """
    import pydicom

    ds = pydicom.dcmread(dicom_path)
    if not hasattr(ds, "PixelData"):
        print(f"[WARNING] No pixel data in file: {dicom_path}")
        return None, {"file_path": dicom_path, "error": "No PixelData"}

    # Determine frame rate from several possible tags
    frame_rate = 15
    frame_rate_tags = [(0x08, 0x2144), (0x18, 0x1063), (0x18, 0x40), (0x7FDF, 0x1074)]
    for tag in frame_rate_tags:
        try:
            frame_rate = float(ds[tag].value)
            break
        except (KeyError, AttributeError):
            pass
    if data_type != "ANGIO" and frame_rate == 15:
        frame_rate = 30

    # Get photometric interpretation (defaulting to MONOCHROME2)
    photo_type = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    pixel_array = ds.pixel_array

    # Determine if the image is multi-frame:
    # - If the array is 3D and the last dimension is 3, treat it as a single RGB image.
    # - Otherwise, if a NumberOfFrames attribute exists and > 1, treat it as multi-frame.
    if pixel_array.ndim == 3 and pixel_array.shape[-1] == 3:
        frames = [pixel_array]
    else:
        num_frames = int(getattr(ds, "NumberOfFrames", 1))
        if num_frames <= 1:
            frames = [pixel_array]
        else:
            frames = [frame for frame in pixel_array]

    processed_frames = []
    for frame in frames:
        if photo_type == "MONOCHROME2":
            # For grayscale, convert to 3-channel image.
            if frame.ndim == 2:
                proc = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                proc = cv2.cvtColor(frame[..., 0], cv2.COLOR_GRAY2BGR)
        elif photo_type == "PALETTE COLOR":
            # Retrieve the LUT descriptor (e.g., [256, 0, 16])
            red_desc = ds[0x0028, 0x1101].value
            n_entries, first_index, bits = red_desc

            # Convert the LUT bytes to numpy arrays using frombuffer
            red_lut = np.frombuffer(ds[0x0028, 0x1201].value, dtype=np.uint16)[:n_entries]
            green_lut = np.frombuffer(ds[0x0028, 0x1202].value, dtype=np.uint16)[:n_entries]
            blue_lut = np.frombuffer(ds[0x0028, 0x1203].value, dtype=np.uint16)[:n_entries]

            # Scale LUT values to 8-bit if needed
            if bits > 8:
                factor = 2 ** (bits - 8)
                red_lut = (red_lut / factor).astype(np.uint8)
                green_lut = (green_lut / factor).astype(np.uint8)
                blue_lut = (blue_lut / factor).astype(np.uint8)
            else:
                red_lut = red_lut.astype(np.uint8)
                green_lut = green_lut.astype(np.uint8)
                blue_lut = blue_lut.astype(np.uint8)

            # Create a combined LUT (shape: [n_entries, 3])
            lut = np.stack((red_lut, green_lut, blue_lut), axis=-1)

            # Map the pixel indices to RGB values (adjust for first_index)
            try:
                proc = lut[frame - first_index]
                # Convert from RGB -> BGR for correct OpenCV saving FIXES BLUE VIDEO
                proc = proc[..., ::-1]
            except Exception as e:
                print(f"[WARNING] Palette color mapping failed for {dicom_path} with error: {e}")
                proc = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # fallback to grayscale
        elif photo_type == "RGB":
            if frame.ndim == 2:
                proc = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3:
                # Suppose this results in an RGB array ...
                proc = frame if frame.shape[-1] == 3 else np.transpose(frame, (1, 2, 0))
                # Then flip to BGR:
                proc = proc[..., ::-1]
            else:
                proc = frame
        elif photo_type in ("YBR_FULL", "YBR_FULL_422"):
            if frame.ndim == 2:
                proc = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3:
                proc = frame if frame.shape[-1] == 3 else np.transpose(frame, (1, 2, 0))
            else:
                proc = frame
            try:
                proc = cv2.cvtColor(proc, cv2.COLOR_YCrCb2BGR)
            except Exception as e:
                print(f"[WARNING] Error converting YBR image: {e}")
        else:
            # Default: assume grayscale
            if frame.ndim == 2:
                proc = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                proc = frame

        if proc.dtype != np.uint8:
            proc = cv2.normalize(proc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        processed_frames.append(proc)

    # If only one frame (screenshot), save as PNG; otherwise, build video from frames.
    if len(processed_frames) == 1:
        output_file = output_path.replace(".mp4", ".png")
        cv2.imwrite(output_file, processed_frames[0])
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, frm in enumerate(processed_frames):
                frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                cv2.imwrite(frame_path, frm)
            if lossless:
                ffmpeg_command = [
                    "ffmpeg",
                    "-framerate",
                    str(frame_rate),
                    "-i",
                    os.path.join(temp_dir, "frame_%04d.png"),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-qp",
                    "0",
                    "-y",
                    output_path,
                ]
            else:
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
                    "-preset",
                    preset,
                    "-y",
                    output_path,
                ]
            try:
                subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"FFmpeg command failed: {e.stderr.strip()}")
                raise
        output_file = output_path

    # Convert metadata (excluding large binary fields) to a serializable dict.
    dicom_dict = {}
    for elem in ds.iterall():
        if elem.keyword not in ["PixelData", "WaveformData", "CurveData"]:
            tag_str = f"({elem.tag.group:04x}, {elem.tag.element:04x})"
            if tag_str not in ["(5000, 3000)", "(7fe0, 0010)"]:
                dicom_dict[elem.keyword] = convert_to_serializable(elem.value)
    dicom_dict["video_path"] = output_file
    return output_file, dicom_dict


def _convert_npz_worker(args):
    """Worker function for parallel NPZ processing."""
    input_path, output_path, fps, crf, preset, lossless = args
    try:
        data = np.load(input_path)
        if "pixel_array" not in data:
            return {
                "input_file": input_path,
                "output_file": output_path,
                "status": "error",
                "error_message": "No pixel_array found",
            }
        pixel_array = data["pixel_array"]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if pixel_array.dtype != np.uint8:
            pixel_array = (
                (pixel_array - pixel_array.min()) * 255 / (pixel_array.max() - pixel_array.min())
            ).astype(np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, frame in enumerate(pixel_array):
                frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                cv2.imwrite(frame_path, frame)
            if lossless:
                ffmpeg_command = [
                    "ffmpeg",
                    "-framerate",
                    str(fps),
                    "-i",
                    os.path.join(temp_dir, "frame_%04d.png"),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-qp",
                    "0",
                    "-y",
                    output_path,
                ]
            else:
                ffmpeg_command = [
                    "ffmpeg",
                    "-framerate",
                    str(fps),
                    "-i",
                    os.path.join(temp_dir, "frame_%04d.png"),
                    "-c:v",
                    "libx264",
                    "-crf",
                    str(crf),
                    "-preset",
                    preset,
                    "-y",
                    output_path,
                ]
            result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
            if result.returncode != 0:
                return {
                    "input_file": input_path,
                    "output_file": output_path,
                    "status": "error",
                    "error_message": result.stderr,
                }
        return {
            "input_file": input_path,
            "output_file": output_path,
            "num_frames": len(pixel_array),
            "fps": fps,
            "frame_shape": str(pixel_array[0].shape),
            "compression": "lossless" if lossless else f"crf{crf}",
            "status": "success",
            "error_message": "",
        }
    except Exception as e:
        return {
            "input_file": input_path,
            "output_file": output_path,
            "status": "error",
            "error_message": str(e),
        }


def format_time_column(df, column_name):
    """
    Formats a time column from HHMMSS.decimal or HHMMSS format to HH:MM:SS time format and overwrites the original column.

    Parameters:
    - df: pandas DataFrame containing the column to format.
    - column_name: string, name of the column in the DataFrame to format.

    The function directly modifies the input DataFrame by updating the specified time column to the HH:MM:SS format.
    """
    # Convert the column to string to ensure manipulation is possible
    df[column_name] = df[column_name].astype(str)

    # Remove decimals and any digits following (if present) to ensure a strict HHMMSS format
    no_decimals = (
        df[column_name].str.split(".").str[0].str.pad(width=6, side="left", fillchar="0")
    )

    # Convert to a proper time format (HH:MM:SS), handling errors with 'coerce' to avoid crashes on unexpected formats
    formatted_time = pd.to_datetime(no_decimals, format="%H%M%S", errors="coerce").dt.time

    # Overwrite the original column with the formatted time
    df[column_name] = formatted_time


def convert_npz_batch_to_h264(
    input_df,
    file_path_column,
    output_dir,
    fps=30,
    crf=23,
    preset="medium",
    lossless=False,
    num_processes=None,
):
    """
    Convert batch of NPZ files to H.264 videos in parallel.
    """
    if file_path_column not in input_df.columns:
        raise ValueError(f"Column '{file_path_column}' not found in input DataFrame")
    os.makedirs(output_dir, exist_ok=True)
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    process_args = []
    for _, row in input_df.iterrows():
        input_path = row[file_path_column]
        rel_path = os.path.relpath(
            input_path, "/media/data1/ravram/CoronaryDominance/extracted_data/"
        )
        output_path = os.path.join(output_dir, rel_path).replace(".npz", ".mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        process_args.append((input_path, output_path, fps, crf, preset, lossless))
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(_convert_npz_worker, process_args),
                total=len(process_args),
                desc="Converting NPZ files",
            )
        )
    metadata_df = pd.DataFrame(results)
    metadata_df["input_path"] = [args[0] for args in process_args]
    metadata_df["output_path"] = [args[1] for args in process_args]
    metadata_path = os.path.join(output_dir, "conversion_metadata.csv")

    metadata_df.to_csv(metadata_path, index=False)
    return metadata_df


def extract_h264_and_metadata(
    path,
    data_type="ANGIO",
    destinationFolder="dicom_h264_extracted/",
    dataFolder="data/",
    dicom_path_column="path",
    subdirectory=None,
    num_processes=None,
    lossless=False,
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

    existing_df = None
    if os.path.exists(final_path):
        existing_df = pd.read_csv(final_path)

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            process_row,
            [
                (row, destinationFolder, subdirectory, dicom_path_column, data_type, lossless)
                for _, row in tqdm(df.iterrows())
            ],
        )

    final_list = [json.loads(result) for result in results if result is not None]
    if not final_list:
        print("No valid results were returned from processing.")
        return None

    dicom_df_final = pd.DataFrame(final_list)
    dicom_df_final = process_metadata(dicom_df_final, data_type)

    # Format time columns if they exist
    if "SeriesTime" in dicom_df_final.columns:
        format_time_column(dicom_df_final, "SeriesTime")
    if "StudyTime" in dicom_df_final.columns:
        format_time_column(dicom_df_final, "StudyTime")

    if existing_df is not None:
        # Combine existing and new data, keeping last occurrence of duplicates
        combined_df = pd.concat([existing_df, dicom_df_final])
        dicom_df_final = combined_df.drop_duplicates(subset=["FileName"], keep="last")

    dicom_df_final.to_csv(final_path, index=False)
    print(f"Metadata saved to: {final_path}")

    with open(final_path) as csvfile, open(final_path_alpha, "w", newline="") as alphafile:
        reader = csv.reader(csvfile)
        writer = csv.writer(alphafile, delimiter="α")
        for row in reader:
            writer.writerow(row)
    print(f"Alpha-delimited metadata saved to: {final_path_alpha}")

    return dicom_df_final


def process_row(
    row, destinationFolder, subdirectory, dicom_path_column, data_type="ANGIO", lossless=False
):
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
    try:
        dicom_path = os.path.join(row[dicom_path_column])
    except KeyError:
        # First try case-insensitive match
        if dicom_path_column.lower() in [col.lower() for col in row.index]:
            actual_col = next(
                col for col in row.index if col.lower() == dicom_path_column.lower()
            )
            dicom_path = os.path.join(row[actual_col])
        # Then try 'DICOM File Path' as a fallback
        elif "DICOM File Path" in row.index:
            dicom_path = os.path.join(row["DICOM File Path"])
        # Finally raise error with available columns
        else:
            available_cols = sorted(list(row.index))
            raise KeyError(
                f"Column '{dicom_path_column}' not found in row and 'DICOM File Path' not found. Available columns: {available_cols}"
            )
    output_filename = os.path.basename(dicom_path).replace(".dcm", ".mp4")
    output_path = os.path.join(destinationPath, output_filename)
    _, metadata = extract_h264_video_from_dicom(
        dicom_path, output_path, lossless=lossless, data_type=data_type
    )
    serializable_metadata = {k: convert_to_serializable(v) for k, v in metadata.items()}
    if "file" not in serializable_metadata:
        serializable_metadata["file"] = dicom_path
    if "video_path" not in serializable_metadata:
        serializable_metadata["video_path"] = output_path
    return json.dumps(serializable_metadata)


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract H.264 videos and metadata from DICOM or NPZ files."
    )
    parser.add_argument("--input_file", required=True, help="Path to the input CSV file")
    parser.add_argument(
        "--file_type",
        required=True,
        choices=["dicom", "npz"],
        help="Type of files to process (dicom or npz)",
    )
    parser.add_argument(
        "--file_path_column",
        required=True,
        help="Column name containing file paths (DICOM or NPZ)",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Destination folder for extracted videos"
    )
    parser.add_argument(
        "--metadata_dir", required=True, help="Folder for output metadata CSV files"
    )
    parser.add_argument("--subdirectory", help="Column name for subdirectory information")
    parser.add_argument("--data_type", default="ANGIO", help="Type of DICOM data (e.g., 'ANGIO')")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for output video")
    parser.add_argument(
        "--num_processes",
        type=int,
        help="Number of processes to use (default: number of CPU cores)",
    )
    parser.add_argument(
        "--lossless", action="store_true", help="Use lossless compression for video extraction"
    )
    args = parser.parse_args(args)
    print(f"Processing {args.file_type} files")
    print(f"Input file: {args.input_file}")
    print(f"File path column: {args.file_path_column}")
    print(f"Output directory: {args.output_dir}")
    print(f"Lossless compression: {args.lossless}")
    if args.file_type == "dicom":
        extract_h264_and_metadata(
            args.input_file,
            data_type=args.data_type,
            dicom_path_column=args.file_path_column,
            destinationFolder=args.output_dir,
            subdirectory=args.subdirectory,
            dataFolder=args.metadata_dir,
            num_processes=args.num_processes,
            lossless=args.lossless,
        )
    else:  # npz
        df = pd.read_csv(args.input_file)
        metadata_df = convert_npz_batch_to_h264(
            df,
            args.file_path_column,
            args.output_dir,
            fps=args.fps,
            lossless=args.lossless,
            num_processes=args.num_processes,
        )
        print(f"\nConversion summary:")
        print(f"Total files processed: {len(metadata_df)}")
        print(f"Successful conversions: {len(metadata_df[metadata_df['status'] == 'success'])}")
        print(f"Failed conversions: {len(metadata_df[metadata_df['status'] == 'error'])}")
    print("Done")


if __name__ == "__main__":
    main()
