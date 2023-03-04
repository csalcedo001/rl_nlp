import os
import json

from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube

def download_youtube_video_data(video_id, save_dir):
    # Get captions
    captions = YouTubeTranscriptApi.get_transcript(video_id)

    captions_path = os.path.join(save_dir, 'captions.json')
    with open(captions_path, 'w') as in_file:
        json.dump(captions, in_file, indent=4)
    

    # Get video in mp4 format
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    yt = YouTube(video_url)

    yt.streams \
        .filter(progressive=True, file_extension='mp4') \
        .order_by('resolution') \
        .desc() \
        .first() \
        .download(output_path=save_dir)


### Test it out
video_id = "B-Rqub5ZPec"

save_dir = os.path.join(os.path.dirname(__file__), 'videos')
os.makedirs(save_dir, exist_ok=True)

download_youtube_video_data(video_id, save_dir)