import os
from datetime import timedelta
from moviepy import editor
from PIL import Image
import numpy as np
import speech_recognition as sr

def split_audio_and_frames(video_path, output_audio_path, output_frames_path, fps):
    video_clip = editor.VideoFileClip(video_path)

    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_audio_path)
    # text = audio_to_text(output_audio_path)
    # with open(output_text_path, 'w') as text_file:
    #     text_file.write(text)

    frames = list(video_clip.iter_frames(fps=fps, dtype='uint8'))
    
    os.makedirs(output_frames_path, exist_ok=True)

    frame_timestamps = []
    for i, frame in enumerate(frames):
        timestamp = video_clip.duration * (i / len(frames))
        image = Image.fromarray(np.array(frame))
        image_path = os.path.join(output_frames_path, f"frame_{i:04d}.png")
        image.save(image_path)
        # frame_timestamps.append((timestamp, image))

    video_clip.close()
    audio_clip.close()

    return frame_timestamps

def audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as audio_file:
        audio_data = recognizer.record(audio_file)
        text = recognizer.recognize_google(audio_data)
    return text

if __name__ == "__main__":
    input_video_path = '/home/lenovo/Downloads/testing_2.mp4'
    output_audio_path = '/home/lenovo/mnt/hackathon/Detection-of-Sensitive-Data-Exposure-in-Images/output/output_audio.wav'
    output_frames_path = '/home/lenovo/mnt/hackathon/Detection-of-Sensitive-Data-Exposure-in-Images/output/images'
    output_text_path = '/home/lenovo/mnt/hackathon/Detection-of-Sensitive-Data-Exposure-in-Images/output/output_text.txt'

    fps = 5

    timestamps_and_images = split_audio_and_frames(input_video_path, output_audio_path, output_frames_path, fps)

    for timestamp, image in timestamps_and_images:
        print(f"Timestamp: {str(timedelta(seconds=timestamp))}, Image: {image}")