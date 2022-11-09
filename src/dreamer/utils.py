import cv2
import numpy as np

def save_video(
        frames,
        filename,
        fps=20,
        channel_first=True,
        low=0.,
        high=1.
    ):
    if channel_first:
        frame_size = frames[0].shape[1:]
    else:
        frame_size = frames[0].shape[:2]

    out = cv2.VideoWriter(
        filename,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        frame_size
    )

    for frame in frames:
        if channel_first:
            frame = np.transpose(frame, (1, 2, 0))

        frame = (frame - low) / (high - low) * 255.
        frame = frame.astype(np.uint8)

        out.write(frame)

    out.release()