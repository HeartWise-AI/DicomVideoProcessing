# Notebook which iterates through a folder, including subfolders,
# and convert DICOM files to AVI files
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
        # If it's large (e.g., waveforms), skip or truncate
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
        # If array is large, truncate
        if obj.size > max_length:
            return f"<large ndarray, shape={obj.shape}, dtype={obj.dtype}>"
        else:
            return obj.tolist()
    elif isinstance(obj, DicomSequence):
        return [convert_to_serializable(ds, max_length) for ds in obj]
    elif isinstance(obj, DicomDataset):
        # Convert each element except PixelData, CurveData, or waveforms
        serial_dict = {}
        for elem in obj.iterall():
            tag_str = f"({elem.tag.group:04x}, {elem.tag.element:04x})"
            # Skip PixelData or giant waveforms
            if elem.keyword in ["PixelData", "WaveformData", "CurveData"]:
                continue
            # Some curve data or large binary arrays
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
    # Drop FrameTimeVector column if it exists
    if "FrameTimeVector" in processed_metadata.columns:
        processed_metadata = processed_metadata.drop(columns=["FrameTimeVector"])

    return processed_metadata


def extract_h264_video_from_dicom(
    dicom_path, output_path, crf=23, preset="medium", data_type="ANGIO", lossless=False
):
    """Read DICOM, extract frames to PNG, then encode H.264 with FFmpeg. Return metadata (minus large fields)."""
    import pydicom

    dicom_data = pydicom.dcmread(dicom_path)
    
    # If no pixel data, skip
    if not hasattr(dicom_data, "PixelData"):
        print(f"[WARNING] No pixel data in file: {dicom_path}")
        return None, {"file_path": dicom_path, "error": "No PixelData"}
    pixel_array = dicom_data.pixel_array

    # If pixel_array is not a 3D array, skip
    if len(pixel_array.shape) != 3:
        print(f"[WARNING] Not a movie: {dicom_path}")
        return None, {"file_path": dicom_path, "error": "Not a movie"}
    
    # If no frame rate, skip
    if not (0x08, 0x2144) in dicom_data:
        print(f"[WARNING] No frame rate in DICOM: {dicom_path}")
        return None, {"file_path": dicom_path, "error": "No frame rate in DICOM"}

    # If frame rate is too low or video is too short, skip
    frame_rate = dicom_data[(0x08, 0x2144)].value
    # all dicom with a fps lower of 5 are bad
    if frame_rate < 5 or frame_rate > pixel_array.shape[0]: # at least 1 second of video
        print(f"[WARNING] Frame rate is too low: {frame_rate} with {pixel_array.shape[0]} frames for {dicom_path}")
        return None, {"file_path": dicom_path, "error": "Frame rate is too low"}

    # Stretch pixel range to [0, 255]
    pixel_array = pixel_array.astype(np.float32)
    pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
    pixel_array = (pixel_array * 255).astype(np.uint8) # need to be uint8 for cv2

    # Encode video as AVI
    try:
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # FFV1 codec - lossless
        out = cv2.VideoWriter(
            output_path,
            fourcc, 
            frame_rate, 
            (pixel_array.shape[1], pixel_array.shape[2]),
            isColor=False # grayscale frames
        )
        for frame in pixel_array:
            out.write(frame)
        out.release()
    except Exception as e:
        print(f"Video encoding failed: {e}")
        return None, {"file_path": dicom_path, "error": "Video encoding failed"}

    # Convert DICOM metadata to serializable dictionary, skipping large/binary fields
    dicom_dict = {}
    for elem in dicom_data.iterall():
        if elem.keyword not in ["PixelData", "WaveformData", "CurveData"]:
            tag_str = f"({elem.tag.group:04x}, {elem.tag.element:04x})"
            if tag_str not in ["(5000, 3000)", "(7fe0, 0010)"]:
                dicom_dict[elem.keyword] = convert_to_serializable(elem.value)

    dicom_dict["video_path"] = output_path
    return output_path, dicom_dict


def _convert_npz_worker(args):
    """Worker function for parallel processing"""
    input_path, output_path, fps, crf, preset, lossless = args
    try:
        # Load NPZ file
        data = np.load(input_path)
        if "pixel_array" not in data:
            return {
                "input_file": input_path,
                "output_file": output_path,
                "status": "error",
                "error_message": "No pixel_array found",
            }

        pixel_array = data["pixel_array"]

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Normalize pixel values to uint8 if needed
        if pixel_array.dtype != np.uint8:
            pixel_array = (
                (pixel_array - pixel_array.min()) * 255 / (pixel_array.max() - pixel_array.min())
            ).astype(np.uint8)

        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save frames as temporary PNG files
            for i, frame in enumerate(pixel_array):
                frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                cv2.imwrite(frame_path, frame)

            # Construct FFmpeg command
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

            # Run FFmpeg
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

    Args:
        input_df (pd.DataFrame): DataFrame containing file paths
        file_path_column (str): Name of column containing NPZ file paths
        output_dir (str): Directory to save output videos
        fps (int): Frames per second for output videos
        crf (int): Constant Rate Factor for H.264 compression
        preset (str): FFmpeg preset
        lossless (bool): Whether to use lossless compression
        num_processes (int): Number of parallel processes to use
    """
    if file_path_column not in input_df.columns:
        raise ValueError(f"Column '{file_path_column}' not found in input DataFrame")

    os.makedirs(output_dir, exist_ok=True)

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    # Prepare arguments for parallel processing
    process_args = []
    for _, row in input_df.iterrows():
        input_path = row[file_path_column]
        # Example: preserve directory structure if needed
        rel_path = os.path.relpath(
            input_path, "/media/data1/ravram/CoronaryDominance/extracted_data/"
        )
        output_path = os.path.join(output_dir, rel_path).replace(".npz", ".mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        process_args.append((input_path, output_path, fps, crf, preset, lossless))

    # Process files in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(_convert_npz_worker, process_args),
                total=len(process_args),
                desc="Converting NPZ files",
            )
        )

    # Create metadata DataFrame
    metadata_df = pd.DataFrame(results)

    # Add input and output paths to metadata
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
    sep="α"
):
    from tqdm import tqdm

    try:
        df = pd.read_csv(path, sep=sep, engine="python")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Empty data in {path}")

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
                (row, destinationFolder, subdirectory, dicom_path_column, data_type, lossless)
                for _, row in tqdm(df.iterrows(), desc="Preprocessing rows", total=len(df))
            ],
        )
    final_list = [json.loads(result) for result in results if result is not None]

    if not final_list:
        print("No valid results were returned from processing.")
        return None

    dicom_df_final = pd.DataFrame(final_list)
    dicom_df_final = process_metadata(dicom_df_final, data_type)

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
        # Check if the column name is case sensitive
        if dicom_path_column.lower() in [col.lower() for col in row.index]:
            # Find the actual column name with correct case
            actual_col = next(
                col for col in row.index if col.lower() == dicom_path_column.lower()
            )
            dicom_path = os.path.join(row[actual_col])
        else:
            raise KeyError(
                f"Column '{dicom_path_column}' not found in row. Available columns: {list(row.index)}"
            )

    output_filename = os.path.basename(dicom_path).replace(".dcm", ".avi")
    output_path = os.path.join(destinationPath, output_filename)

    # Extract H.264 video and get metadata
    _, metadata = extract_h264_video_from_dicom(
        dicom_path, output_path, lossless=lossless, data_type=data_type
    )

    # Convert metadata to serializable format
    serializable_metadata = {k: convert_to_serializable(v) for k, v in metadata.items()}

    # Add file and video_path to metadata if not already present
    if "file" not in serializable_metadata:
        serializable_metadata["file"] = dicom_path
    if "video_path" not in serializable_metadata:
        serializable_metadata["video_path"] = output_path

    # Return a JSON string version
    return json.dumps(serializable_metadata)


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract H.264 videos and metadata from DICOM or NPZ files."
    )

    # Input arguments
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
        "--sep",
        required=False,
        default="α",
        help="Separator for the input file (default: α)",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir", required=True, help="Destination folder for extracted videos"
    )
    parser.add_argument(
        "--metadata_dir", required=True, help="Folder for output metadata CSV files"
    )
    parser.add_argument("--subdirectory", help="Column name for subdirectory information")

    # Processing options
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
    print(f"Separator: {args.sep}")

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
            sep=args.sep
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
