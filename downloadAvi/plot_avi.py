import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom as dicom
import zarr
from tqdm import tqdm


def normalize_image(image):
    """Normalize image to have pixel values between 0 and 255."""
    image_min, image_max = image.min(), image.max()
    # Avoid division by zero if image is already constant
    if image_max == image_min:
        return np.zeros_like(image, dtype=np.uint8)
    normalized_image = (image - image_min) / (image_max - image_min) * 255
    return normalized_image.astype(np.uint8)


def sample_and_plot_middle_frames(
    df, N, label_column=None, second_label_column=None, path_column=None
):
    """
    Randomly samples N filenames from df[path_column] and plots the middle frame of each
    (or the image itself if single-frame).

    Args:
        df (DataFrame): DataFrame with file paths in path_column.
        N (int): Number of items to sample and plot.
        label_column (str): Column name in df for the main label (optional).
        second_label_column (str): Column name in df for a second label (optional).
        path_column (str): Column name in df that has the file paths to read.

    Returns:
        None
    """
    sampled_filenames = df[path_column].sample(N, replace=len(df) < N)

    # Calculate rows & cols for subplots
    num_rows = (N + 4) // 5
    num_cols = min(N, 5)

    # Convert label columns to string in case they're numeric
    if label_column and label_column in df.columns:
        df[label_column] = df[label_column].astype(str)
    if second_label_column and second_label_column in df.columns:
        df[second_label_column] = df[second_label_column].astype(str)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))

    # If only 1 subplot, make it iterable
    if N == 1:
        axes = np.array([axes])

    for i, filename in enumerate(sampled_filenames):
        # Subplot indexing
        row_idx = i // num_cols
        col_idx = i % num_cols
        ax = axes[row_idx][col_idx] if num_rows > 1 else axes[col_idx]

        # Initialize variables
        frame_rgb = None
        title_text = "Frame"

        # ---- 1) DICOM (.dcm) ----
        if filename.lower().endswith(".dcm"):
            ds = dicom.dcmread(filename)

            # Extract pixel array
            image_array = ds.pixel_array

            # If multi-frame, pick middle; else single
            total_frames = getattr(ds, "NumberOfFrames", 1)
            if total_frames > 1:
                mid_idx = total_frames // 2
                frame = image_array[mid_idx]
            else:
                frame = image_array

            # If 16-bit, normalize
            if frame.dtype == np.uint16:
                frame = normalize_image(frame)

            # Convert grayscale → RGB if needed
            if frame.ndim == 2:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                # If it's already 3D, assume it’s some color format
                frame_rgb = frame
            title_text = ".dcm"

        # ---- 2) Standard image files (PNG, JPG, etc.) ----
        elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_bgr = cv2.imread(filename)
            if img_bgr is None:
                print(f"Warning: Unable to read image file: {filename}")
                continue
            frame_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            title_text = "Image"

        # ---- 3) Video files: .avi or .mp4 (or .mov, .mkv, etc. if you like) ----
        elif filename.lower().endswith((".avi", ".mp4")):
            cap = cv2.VideoCapture(filename)
            if not cap.isOpened():
                print(f"Warning: Unable to open video file: {filename}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < 1:
                print(f"Warning: Video has no frames: {filename}")
                cap.release()
                continue

            middle_frame_number = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_number)
            ret, frame_bgr = cap.read()
            cap.release()

            if not ret or frame_bgr is None:
                print(f"Warning: Unable to read middle frame from video: {filename}")
                continue

            # Convert from BGR to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            title_text = "Video"

        # ---- 4) Zarr (.zarr) data (video or single volume) ----
        elif filename.lower().endswith(".zarr"):
            data = zarr.open(filename, mode="r")
            if "video" not in data:
                print(f"Warning: 'video' dataset not found in zarr file: {filename}")
                continue
            vid = np.array(data["video"])  # shape might be (Frames, H, W) or (Frames, H, W, C)

            # Handle grayscale vs color
            if vid.ndim == 3:
                # (F, H, W) → expand to (F, H, W, 3)
                vid = np.stack([vid] * 3, axis=-1)
            elif vid.ndim == 4 and vid.shape[-1] != 3:
                # If last dim >3, take first 3 channels
                vid = vid[..., :3]

            # Grab middle frame
            mid_idx = vid.shape[0] // 2
            frame_rgb = vid[mid_idx]

            # Normalize if not uint8
            if frame_rgb.dtype != np.uint8:
                frame_rgb = normalize_image(frame_rgb)

            title_text = ".zarr"

        else:
            # Unknown extension: skip
            print(f"Warning: Unhandled file extension for {filename}")
            continue

        # If we still have no frame, skip
        if frame_rgb is None:
            continue

        # Make sure shape is (H, W, 3)
        if frame_rgb.ndim == 2:
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2RGB)
        elif frame_rgb.shape[-1] != 3:
            # If the last dimension is not 3, force 3 channels
            frame_rgb = np.repeat(frame_rgb[..., np.newaxis], 3, axis=-1)

        # Plot
        ax.imshow(frame_rgb)
        ax.axis("off")

        # Build up the title with any label columns
        if label_column and label_column in df.columns:
            label_val = df.loc[df[path_column] == filename, label_column].values[0]
            title_text += f" | {label_val}"

        if second_label_column and second_label_column in df.columns:
            second_label_val = df.loc[df[path_column] == filename, second_label_column].values[0]
            title_text += f" - {second_label_val}"

        # Set subplot title
        ax.set_title(title_text)

    plt.tight_layout()
    plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Sample and plot middle frames from videos/images."
    )
    parser.add_argument(
        "--input_file", type=str, help="Path to the input CSV file containing paths."
    )
    parser.add_argument("--no_frames", type=int, default=5, help="Number of items to sample.")
    parser.add_argument("--label_column", type=str, help="Column name for labels.")
    parser.add_argument("--second_label_column", type=str, help="Second column name for labels.")
    parser.add_argument(
        "--path_column", type=str, required=True, help="Column name for the file paths."
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    sample_and_plot_middle_frames(
        df,
        args.no_frames,
        label_column=args.label_column,
        second_label_column=args.second_label_column,
        path_column=args.path_column,
    )
    print("Done")


if __name__ == "__main__":
    main()
