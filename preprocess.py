"""
Data Preprocessing for Mobile Snatching Detection
Handles: Frame Extraction, Normalization, and Cropping
"""

import cv2
import numpy as np
import os
from tqdm import tqdm


def normalize_frame(frame):
    """
    Min-max normalization as per Equation (1) in the paper:
        I_normalized(x, y) = (I(x,y) - I_min) / (I_max - I_min)

    Args:
        frame (np.ndarray): Raw input frame

    Returns:
        np.ndarray: Normalized frame with pixel values in [0, 1]
    """
    i_min = frame.min()
    i_max = frame.max()
    if i_max - i_min == 0:
        return frame.astype(np.float32)
    return (frame - i_min) / (i_max - i_min)


def crop_frame(frame, x1, y1, x2, y2):
    """
    Crop frame to region of interest as per Equation (2):
        C = I[X1, X2 : Y1, Y2]

    Args:
        frame: Input frame
        x1, y1: Top-left corner
        x2, y2: Bottom-right corner

    Returns:
        Cropped frame
    """
    return frame[y1:y2, x1:x2]


def extract_frames(video_path, num_frames=64, img_size=(240, 240)):
    """
    Extract a fixed number of frames from a video.

    Rules (from paper Section 3.1.1):
      - Target: 64 frames
      - If video has fewer than 64 frames: pad with first/last frames
      - If video has more than 160 frames: sample every 3rd frame

    Args:
        video_path (str): Path to input video file
        num_frames (int): Target number of frames. Default: 64
        img_size (tuple): Resize dimensions (H, W). Default: (240, 240)

    Returns:
        np.ndarray: Array of shape (num_frames, H, W, 3)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, img_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    total = len(frames)

    if total == 0:
        raise ValueError(f"Could not read any frames from: {video_path}")

    # Sampling strategy
    if total > 160:
        frames = frames[::3]  # Every 3rd frame
    
    frames = np.array(frames, dtype=np.float32)

    # Pad if too short
    if len(frames) < num_frames:
        pad_start = [frames[0]] * ((num_frames - len(frames)) // 2)
        pad_end = [frames[-1]] * (num_frames - len(frames) - len(pad_start))
        frames = np.array(pad_start + list(frames) + pad_end, dtype=np.float32)

    # Trim to target length
    if len(frames) > num_frames:
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        frames = frames[indices]

    # Normalize each frame
    frames = np.array([normalize_frame(f) for f in frames])

    return frames  # shape: (num_frames, H, W, 3)


def preprocess_dataset(input_dir, output_dir, num_frames=64, img_size=(240, 240)):
    """
    Preprocess all videos in a dataset directory.

    Expected directory structure:
        input_dir/
            snatching/  ← snatching videos
            normal/     ← normal videos

    Args:
        input_dir (str): Path to raw video directory
        output_dir (str): Path to save processed numpy arrays
        num_frames (int): Frames per video. Default: 64
        img_size (tuple): Frame resize dimensions. Default: (240, 240)
    """
    os.makedirs(output_dir, exist_ok=True)
    classes = ['snatching', 'normal']

    for label, cls in enumerate(classes):
        cls_input = os.path.join(input_dir, cls)
        cls_output = os.path.join(output_dir, cls)
        os.makedirs(cls_output, exist_ok=True)

        videos = [f for f in os.listdir(cls_input)
                  if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

        print(f"\nProcessing class '{cls}' ({len(videos)} videos)...")
        for vid in tqdm(videos):
            vid_path = os.path.join(cls_input, vid)
            try:
                frames = extract_frames(vid_path, num_frames, img_size)
                save_path = os.path.join(cls_output, vid.rsplit('.', 1)[0] + '.npy')
                np.save(save_path, frames)
            except Exception as e:
                print(f"  ⚠ Skipped {vid}: {e}")

    print("\n✅ Preprocessing complete.")


if __name__ == '__main__':
    preprocess_dataset(
        input_dir='data/raw_videos/',
        output_dir='data/processed/',
        num_frames=64,
        img_size=(240, 240)
    )
