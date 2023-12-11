import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sample_and_plot_middle_frames(df, N):
    # Randomly sample N filenames
    sampled_filenames = df["FileName"].sample(N)

    # Set up the plot
    fig, axes = plt.subplots(1, N, figsize=(N * 5, 5))

    for i, filename in enumerate(sampled_filenames):
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

            # Plotting the frame
            axes[i].imshow(frame_rgb)
            axes[i].axis("off")
            axes[i].set_title(f"Frame {middle_frame_number}")

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

# %%
