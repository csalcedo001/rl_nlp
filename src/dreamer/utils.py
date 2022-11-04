import cv2
import numpy as np

def save_video(frames, filename, fps=20):
    frame_size = frames[0].shape[1:]

    out = cv2.VideoWriter(
        filename,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        frame_size
    )

    for frame in frames:
        frame = np.transpose(frame, (1, 2, 0))
        frame = frame.astype(np.uint8)

        out.write(frame)

    out.release()