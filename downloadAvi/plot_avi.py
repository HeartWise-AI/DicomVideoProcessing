import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sample_and_plot_middle_frames(df, N, label_column=None, second_label_column=None):
    """
    Randomly samples N filenames from the DataFrame and plots the middle frames of the corresponding videos.

    Args:
        df: The DataFrame containing the filenames.
        N: The number of filenames to sample.
        label_column: The column in the DataFrame containing labels (optional).
        second_label_column: The second column in the DataFrame containing labels (optional).

    Returns:
        None

    Raises:
        None

    Examples:
        sample_and_plot_middle_frames(df, 5, label_column='Label1', second_label_column='Label2')
    """

    sampled_filenames = df["FileName"].sample(N)

    # Calculate the number of rows and columns for the subplots
    num_rows = (N + 4) // 5
    num_cols = min(N, 5)

    # Set up the plot
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
    if N == 1:
        axes = [axes]  # Make it iterable for consistency

    for i, filename in enumerate(sampled_filenames):
        # Calculate the row and column index for the current subplot
        row_idx = i // num_cols
        col_idx = i % num_cols

        # Read the video file
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

        if ret:
            # Convert the frame from BGR to RGB (as Matplotlib displays in RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get the subplot reference
            ax = axes[row_idx][col_idx] if num_rows > 1 else axes[col_idx]

            # Plotting the frame
            ax.imshow(frame_rgb)
            ax.axis("off")

            # Set the title based on provided label columns
            title = f"Frame {middle_frame_number}"
            if label_column:
                subtitle = df.loc[df["FileName"] == filename, label_column].values[0]
                title = subtitle

            if second_label_column:
                second_subtitle = df.loc[df["FileName"] == filename, second_label_column].values[
                    0
                ]
                title += f" - {second_subtitle}"

            ax.set_title(title)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(description="Predictions.")

    parser.add_argument("--input_file")
    parser.add_argument("--no_frames")

    parser = parser.parse_args(args)

    df = pd.read_csv(parser.input_file)

    sample_and_plot_middle_frames(
        parser.input_file,
        parser.no_frames,
    )
    print("Done")


if __name__ == "__main__":
    main()
