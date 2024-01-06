from typing import Union
import subprocess

import cv2
import numpy as np
import torch
from loguru import logger

from models import audio, Landmark_generator, Renderer


def read_video(video_path: str, temp_dir: str):
    fps = 25                                                                    # FIXME: This has not been tested.
    if video_path.split('.')[1] in ['jpg', 'png', 'jpeg']:                      # If input is an image.
        ori_background_frames = [cv2.imread(video_path)]
    else:
        video_stream = cv2.VideoCapture(video_path)
        fps: int = video_stream.get(cv2.CAP_PROP_FPS)
        # FIXME: It might be unnecessary to downsample the video to 25 FPS.
        if fps != 25:
            logger.warning(f'Input video with FPS: {fps}, converting to 25fps...')
            command = f'ffmpeg -y -i {video_path} -r 25 -qp 0 {temp_dir}/temp_25fps.mp4'
            return_code = subprocess.call(command, shell=True)
            if return_code != 0:
                raise RuntimeError(f'ffmpeg command failed with return code {return_code}.')
            video_path = f'{temp_dir}/temp_25fps.mp4'
            video_stream.release()
            video_stream = cv2.VideoCapture(video_path)
            fps: int = video_stream.get(cv2.CAP_PROP_FPS)
        assert fps == 25

        # Input videos frames (includes background as well as face)
        ori_background_frames: list[np.ndarray] = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            ori_background_frames.append(frame)
    logger.debug(f'Input video has {len(ori_background_frames)} frames at {fps} FPS.')
    return ori_background_frames, fps


def read_audio(audio_path: str, temp_dir: str, fps: int, mel_step_size: int = 16):
    if not audio_path.endswith('.wav'):
        command = f'ffmpeg -y -i {audio_path} -strict -2 {temp_dir}/temp.wav'
        subprocess.call(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        audio_path = f'{temp_dir}/temp.wav'
    # FIXME: Sample rate is hard coded here.
    wav = audio.load_wav(audio_path, 16000)                                     # Will be resampled to 16000 Hz.
    mel = audio.melspectrogram(wav)                                             # (H, W)

    # NOTE: Number of mel chunks per second is 80. At 25 FPS, 5 frames = 200 ms, so the mel_step_size is 16.
    # NOTE: Thus, each mel chunk corresponds to 5 frames (200 ms) of the input video.
    # NOTE: Each mel chunk corresponds to 1 frame of the output video.
    mel_chunks: list[np.ndarray] = []
    mel_idx_multiplier = 80. / fps
    mel_chunk_idx = 0
    while 1:
        start_idx = int(mel_chunk_idx * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        mel_chunk_idx += 1
    # mel_chunks = mel_chunks[:(len(mel_chunks) // T) * T]                      # No idea WTF is this.
    logger.debug(f'Input audio with length {len(wav) / 16000:.2f} seconds and {mel.shape[1]} mel frames.')
    logger.debug(f'Input audio with {len(mel_chunks)} chunks.')
    return mel_chunks


def _load(checkpoint_path: str, device: str = 'cuda'):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(model: Union[Landmark_generator, Renderer], path: str, device: str = 'cuda'):
    logger.info(f'Loading model from {path}...')
    checkpoint = _load(path, device)
    s: dict[str, ] = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        if k[:6] == 'module':
            new_k = k.replace('module.', '', 1)
        else:
            new_k = k
        new_s[new_k] = v
    model.load_state_dict(new_s)
    model = model.to(device)
    return model.eval()
