import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom as dicom
from tqdm import tqdm


def normalize_image(image):
    """Normalize image to have pixel values between 0 and 255."""
    image_min, image_max = image.min(), image.max()
    normalized_image = (image - image_min) / (image_max - image_min) * 255
    return normalized_image.astype(np.uint8)


def sample_and_plot_middle_frames(
    df, N, label_column=None, second_label_column=None, path_column=None
):
    """
    Randomly samples N filenames from the DataFrame and plots the middle frames of the corresponding videos.

    Args:
        df: The DataFrame containing the filenames.
        N: The number of filenames to sample.
        label_column: The column in the DataFrame containing labels (optional).
        second_label_column: The second column in the DataFrame containing labels (optional).
        path_column: The column in the DataFrame containing the file paths.

    Returns:
        None

    Raises:
        None

    Examples:
        sample_and_plot_middle_frames(df, 5, label_column='Label1', second_label_column='Label2')
    """

    sampled_filenames = df[path_column].sample(N, replace=True)
    # Calculate the number of rows and columns for the subplots
    num_rows = (N + 4) // 5
    num_cols = min(N, 5)
    df[label_column] = df[label_column].astype(str)
    if second_label_column:
        df[second_label_column] = df[second_label_column].astype(str)

    # Set up the plot
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
    if N == 1:
        axes = [axes]  # Make it iterable for consistency

    for i, filename in enumerate(sampled_filenames):
        # Calculate the row and column index for the current subplot
        row_idx = i // num_cols
        col_idx = i % num_cols
        print(filename)

        if filename.endswith(".dcm"):
            # Load DICOM file and extract the middle frame
            ds = pydicom.dcmread(filename)
            image_array = ds.pixel_array
            total_frames = len(image_array)
            middle_frame_index = total_frames // 2

            # Check if the pixel data is in 16-bit format and normalize if necessary
            if image_array.dtype == np.uint16:
                # Normalize the DICOM image
                image_array = normalize_image(image_array)
            frame = image_array[middle_frame_index] if total_frames > 1 else ds.pixel_array
            # Convert grayscale to 3-channel RGB if necessary
            frame_rgb = (
                cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) if len(frame.shape) == 2 else frame
            )
        elif filename.endswith(".avi"):
            # Read the video file using cv2
            cap = cv2.VideoCapture(filename)

            # Get the total number of frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calculate the middle frame number
            middle_frame_number = total_frames // 2

            # Set the current frame position to the middle frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_number)

            # Read the middle frame
            ret, frame = cap.read()

            # Close the video file
            cap.release()

            if not ret:
                continue

            # Convert the frame from BGR to RGB (as Matplotlib displays in RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            continue
        # Get the subplot reference
        ax = axes[row_idx][col_idx] if num_rows > 1 else axes[col_idx]
        print(frame_rgb.shape)
        # Ensure frame_rgb is in a valid shape for imshow, especially for DICOM video frames
        if (
            frame_rgb.ndim == 3 and frame_rgb.shape[-1] != 3
        ):  # If shape is (F, M, N, C) and C is not 3, convert to RGB
            print(frame_rgb.shape)
            frame_rgb = np.stack([frame_rgb] * 3, axis=-1)  # Expanding to 3 channels
            print(frame_rgb.shape)

        # Plotting the frame
        ax.imshow(frame_rgb)
        ax.axis("off")

        # Initialize title with default value
        title = "Frame"

        # Append middle frame number for .avi files, ensuring variable is defined
        if filename.endswith(".avi") and "middle_frame_number" in locals():
            title += f" {middle_frame_number}"

        # Update title with label from label_column if it exists
        if label_column:
            subtitle = df.loc[df[path_column] == filename, label_column].values[0]
            title = subtitle

        # Append second label from second_label_column if it exists
        if second_label_column:
            second_subtitle = df.loc[df[path_column] == filename, second_label_column].values[0]
            title += f" - {second_subtitle}"

        # Set the subplot title
        ax.set_title(title)


import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom


def normalize_image(image):
    """Normalize image to have pixel values between 0 and 255."""
    image_min, image_max = image.min(), image.max()
    normalized_image = (image - image_min) / (image_max - image_min) * 255
    return normalized_image.astype(np.uint8)


def sample_and_plot_middle_frames(
    df, N, label_column=None, second_label_column=None, path_column=None
):
    """
    Randomly samples N filenames from the DataFrame and plots the middle frames of the corresponding videos.

    Args:
        df: The DataFrame containing the filenames.
        N: The number of filenames to sample.
        label_column: The column in the DataFrame containing labels (optional).
        second_label_column: The second column in the DataFrame containing labels (optional).
        path_column: The column in the DataFrame containing the file paths.

    Returns:
        None

    Raises:
        None

    Examples:
        sample_and_plot_middle_frames(df, 5, label_column='Label1', second_label_column='Label2')
    """

    sampled_filenames = df[path_column].sample(N, replace=True)
    # Calculate the number of rows and columns for the subplots
    num_rows = (N + 4) // 5
    num_cols = min(N, 5)
    df[label_column] = df[label_column].astype(str)
    if second_label_column:
        df[second_label_column] = df[second_label_column].astype(str)

    # Set up the plot
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
    if N == 1:
        axes = [axes]  # Make it iterable for consistency

    for i, filename in enumerate(sampled_filenames):
        # Calculate the row and column index for the current subplot
        row_idx = i // num_cols
        col_idx = i % num_cols
        print(filename)

        if filename.endswith(".dcm"):
            # Load DICOM file and extract the middle frame
            ds = pydicom.dcmread(filename)
            image_array = ds.pixel_array
            total_frames = len(image_array)
            middle_frame_index = total_frames // 2

            # Check if the pixel data is in 16-bit format and normalize if necessary
            if image_array.dtype == np.uint16:
                # Normalize the DICOM image
                image_array = normalize_image(image_array)
            frame = image_array[middle_frame_index] if total_frames > 1 else ds.pixel_array
            # Convert grayscale to 3-channel RGB if necessary
            frame_rgb = (
                cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) if len(frame.shape) == 2 else frame
            )
        elif filename.endswith(".avi"):
            # Read the video file using cv2
            cap = cv2.VideoCapture(filename)

            # Get the total number of frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calculate the middle frame number
            middle_frame_number = total_frames // 2

            # Set the current frame position to the middle frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_number)

            # Read the middle frame
            ret, frame = cap.read()

            # Close the video file
            cap.release()

            if not ret:
                continue

            # Convert the frame from BGR to RGB (as Matplotlib displays in RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            continue
        # Get the subplot reference
        ax = axes[row_idx][col_idx] if num_rows > 1 else axes[col_idx]
        print(frame_rgb.shape)
        # Ensure frame_rgb is in a valid shape for imshow, especially for DICOM video frames
        if (
            frame_rgb.ndim == 3 and frame_rgb.shape[-1] != 3
        ):  # If shape is (F, M, N, C) and C is not 3, convert to RGB
            print(frame_rgb.shape)
            frame_rgb = np.stack([frame_rgb] * 3, axis=-1)  # Expanding to 3 channels
            print(frame_rgb.shape)

        # Plotting the frame
        ax.imshow(frame_rgb)
        ax.axis("off")

        # Initialize title with default value
        title = "Frame"

        # Append middle frame number for .avi files, ensuring variable is defined
        if filename.endswith(".avi") and "middle_frame_number" in locals():
            title += f" {middle_frame_number}"

        # Update title with label from label_column if it exists
        if label_column:
            subtitle = df.loc[df[path_column] == filename, label_column].values[0]
            title = subtitle

        # Append second label from second_label_column if it exists
        if second_label_column:
            second_subtitle = df.loc[df[path_column] == filename, second_label_column].values[0]
            title += f" - {second_subtitle}"

        # Set the subplot title
        ax.set_title(title)


# Define a main function or another appropriate function to handle command-line arguments if necessary
def main():
    # Example of handling command-line arguments, adjust according to your needs
    import argparse

    parser = argparse.ArgumentParser(description="Sample and plot middle frames from videos.")
    parser.add_argument(
        "--input_file", type=str, help="Path to the input CSV file containing video paths."
    )
    parser.add_argument(
        "--no_frames", type=int, default=5, help="Number of frames to sample and plot."
    )
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
