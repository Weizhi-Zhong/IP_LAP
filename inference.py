import argparse
import os
import subprocess

import cv2
import face_alignment
import mediapipe as mp
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from draw_landmark import draw_landmarks
from models import Landmark_generator, Renderer
from util import (MediapipeLandmark, draw_reference_sketches,
                  get_facial_landmarks, load_model,
                  random_select_reference_landmarks, read_audio, read_video)
from util.facemesh import FACEMESH_CONNECTION

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '--input_template_video', type=str)
    parser.add_argument('--audio', type=str)
    parser.add_argument('--output_dir', type=str, default='./test_result')
    parser.add_argument('--temp_dir', type=str, default='./.cache')
    parser.add_argument('--landmark_gen_checkpoint_path', type=str,
                        default='./test/checkpoints/landmarkgenerator_checkpoint.pth')
    parser.add_argument('--renderer_checkpoint_path', type=str,
                        default='./test/checkpoints/renderer_checkpoint.pth')
    parser.add_argument('--maximize_ref', action='store_true')
    # FIXME: The author of the original repo did not implement this feature.
    parser.add_argument('--static', type=bool, help='whether only use the first frame for inference', default=False)

    # Please refer to https://github.com/Weizhi-Zhong/IP_LAP/blob/main/CVPR2023framework.png.
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--num_ref_images', type=int, default=25)
    parser.add_argument('--num_landmarks_transformer', type=int, help='N_{l} in the paper', default=15)
    parser.add_argument('--mel_step_size', type=int, default=16)
    # Will need to retrain the model if you change the following parameters.
    parser.add_argument('--num_frames_transformer', type=int, help='the t of m_{t} in the paper', default=5)
    args = parser.parse_args()

    # The index of the midpoints of the upper lip and lower lip.
    args.lip_index = [0, 17]

    # The index sequence for facial landmarks detected by mediapipe.
    args.ori_sequence_idx = [
        162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288,
        361, 323, 454, 356, 389,
        70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
        336, 296, 334, 293, 300, 276, 283, 282, 295, 285,
        168, 6, 197, 195, 5,
        48, 115, 220, 45, 4, 275, 440, 344, 278,
        33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7,
        362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382,
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95
    ]
    args.full_face_landmark_sequence = [
        *list(range(0, 4)), *list(range(21, 25)), *list(range(25, 91)),         # upper-half face
        *list(range(4, 21)),                                                    # jaw
        *list(range(91, 131))                                                   # mouth
    ]
    if os.path.isfile(args.input):
        if args.input.split('.')[1] in ['jpg', 'png', 'jpeg']:
            args.static = True
    else:
        raise ValueError(f'Input video {args.input} does not exist.')
    if not os.path.isfile(args.audio):
        raise ValueError(f'Input audio {args.audio} does not exist.')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)
    return args


@torch.no_grad()
def main():
    args = parse_args()

    original_background_frames, fps = read_video(args.input, args.temp_dir)
    mel_chunks = read_audio(args.audio, args.temp_dir, fps, args.mel_step_size)
    args.input_vid_len = len(original_background_frames)
    if args.maximize_ref:
        args.num_ref_images = args.input_vid_len
    logger.debug(f'Using {args.num_ref_images} reference images.')

    # (1) Detect facial landmarks using mediapipe tools and face_alignment.
    logger.info('Detecting facial landmarks from the input video...')
    face_crop_results, all_pose_landmarks, all_content_landmarks, lip_dists = get_facial_landmarks(
        original_background_frames, args.lip_index, args.ori_sequence_idx
    )
    ref_lmarks_pose, ref_lmarks_content, lip_dist_idx = random_select_reference_landmarks(
        all_pose_landmarks, all_content_landmarks, lip_dists, args
    )
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
    ref_imgs, ref_img_sketches = draw_reference_sketches(
        face_crop_results, all_pose_landmarks, all_content_landmarks, lip_dist_idx, drawing_spec, args
    )
    face_align_masks = get_face_masks(original_background_frames, face_crop_results)

    ref_lmarks_pose = ref_lmarks_pose.to(device)                                # (N_{l}, 2, 74)
    ref_lmarks_content = ref_lmarks_content.to(device)                          # (N_{l}, 2, 57)
    ref_imgs = ref_imgs.to(device)                                              # (1, N, 3, H, W)
    ref_img_sketches = ref_img_sketches.to(device)                              # (1, N, 3, H, W)

    # (2) Load models.
    landmark_generator = load_model(
        model=Landmark_generator(T=args.num_frames_transformer, d_model=512, nlayers=4, nhead=4, dim_feedforward=1024),
        path=args.landmark_gen_checkpoint_path,
        device=device
    )
    renderer = load_model(model=Renderer(), path=args.renderer_checkpoint_path, device=device)

    # (3) Prolong the input video to match the audio length.
    # TODO: A-B-B-A loop can be improved.
    input_frame_sequence = torch.arange(len(original_background_frames), dtype=int).tolist()
    num_of_repeat = len(mel_chunks) // len(original_background_frames) + 1
    input_frame_sequence = input_frame_sequence + list(reversed(input_frame_sequence))
    input_frame_sequence = input_frame_sequence * ((num_of_repeat + 1) // 2)
    assert len(input_frame_sequence) >= len(mel_chunks)

    # (4) Prepare output video stream.
    frame_h, frame_w, _ = original_background_frames[0].shape
    min_frame_size = min(frame_h, frame_w)
    out_stream = cv2.VideoWriter(
        f'{args.temp_dir}/result.avi',
        cv2.VideoWriter_fourcc(*'I420'),
        fps,
        (frame_w, frame_h),
    )

    # (5) Prepare tensors.
    img_size = args.image_size
    nl_transformer = args.num_frames_transformer                                # N_{l} in the paper.
    ori_sequence_idx = args.ori_sequence_idx
    full_face_landmark_sequence = args.full_face_landmark_sequence
    T_mel_batch = np.zeros((1, nl_transformer, *mel_chunks[0].shape))
    T_pose = torch.zeros((nl_transformer, 2, 74), pin_memory=True)
    T_target_sketches = torch.zeros((nl_transformer, img_size, img_size, 3), pin_memory=True)
    resized_face_imgs = torch.tensor(
        np.array([cv2.resize(img, (img_size, img_size)) / 255 for img, _ in face_crop_results]),
        dtype=torch.float,
        device=device
    ).permute(0, 3, 1, 2)                                                       # (N, 3, 128, 128)

    # ======================================== Inferencing frame by frame ========================================
    logger.info('Start inferencing...')
    for batch_idx, batch_start_idx in tqdm(enumerate(range(0, len(mel_chunks) - 2, 1)), total=len(mel_chunks) - 2):
        # NOTE: Length of mel_chunks is the number of frames of the output video.
        # NOTE: `input_frame_sequence` maps the output frame index to the input frame index. Check (3).
        # NOTE: Take the left 2 and the right 2 chunks from audio as well (nl_transformer == 5).
        # NOTE: Each element in mel_chunks should have the same duration as `nl_transformer * fps`.
        for i, mel_chunk_idx in enumerate(range(batch_start_idx, batch_start_idx + nl_transformer)):
            T_mel_batch[0][i] = mel_chunks[max(0, mel_chunk_idx - 2)]
            input_frame_idx = input_frame_sequence[mel_chunk_idx]
            T_pose[i, :, :] = torch.Tensor(all_pose_landmarks[input_frame_idx]).permute(1, 0)[1:, :]
            if i == 2:
                resized_face_img = resized_face_imgs[input_frame_idx]
                # For post-processing.
                original_frame = original_background_frames[input_frame_idx]
                _, original_face_coordinates = face_crop_results[input_frame_idx]
                face_mask = face_align_masks[input_frame_idx]
        T_pose = T_pose.unsqueeze(0).to(device)                                                     # (1, T, 2, 74)
        T_mels = torch.tensor(T_mel_batch, dtype=torch.float, device=device).unsqueeze(2)           # (1, T, 1, h, w)

        # Landmark generator inference.
        predict_content = landmark_generator(T_mels, T_pose, ref_lmarks_pose, ref_lmarks_content)   # (1 * T, 2, 57)
        T_pose = T_pose.squeeze(0)
        T_predict_full_landmarks = torch.cat([T_pose, predict_content], dim=2).cpu().numpy()        # (1 * T, 2, 131)

        # Draw target sketch.
        for frame_idx in range(nl_transformer):
            full_landmarks = T_predict_full_landmarks[frame_idx]                                    # (2, 131)
            mediapipe_format_landmarks = [
                MediapipeLandmark(
                    ori_sequence_idx[full_face_landmark_sequence[idx]],
                    full_landmarks[0, idx],
                    full_landmarks[1, idx]
                )
                for idx in range(full_landmarks.shape[1])
            ]
            blank = np.zeros((int(frame_h * img_size / min_frame_size), int(frame_w * img_size / min_frame_size), 3))
            drawn_sketech = draw_landmarks(blank, mediapipe_format_landmarks,
                                           connections=FACEMESH_CONNECTION, connection_drawing_spec=drawing_spec)
            drawn_sketech = cv2.resize(drawn_sketech, (img_size, img_size))                     # (128, 128, 3)
            T_target_sketches[frame_idx] = torch.Tensor(drawn_sketech) / 255                    # (T, 128, 128, 3)
        target_sketches = T_target_sketches.permute(0, 3, 1, 2).unsqueeze(0).to(device)         # (1, T, 3, 128, 128)

        # Lower-half masked face.
        ori_face_img = resized_face_img.unsqueeze(0).unsqueeze(0)                               # (1, 1, 3, H, W)

        # Render the full face.
        y1, y2, x1, x2 = original_face_coordinates
        gen_frame = original_frame.copy()
        gen_face, _, _, _ = renderer(ori_face_img, target_sketches,
                                     ref_imgs, ref_img_sketches, T_mels[:, 2].unsqueeze(1))

        # Post-processing.
        gen_face = (gen_face.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # (H, W, 3)
        gen_frame[y1:y2, x1:x2] = cv2.resize(gen_face, (x2 - x1, y2 - y1))
        full = (gen_frame * face_mask + original_frame * (1 - face_mask)).astype(np.uint8)

        if batch_idx == 0:
            out_stream.write(full)
            out_stream.write(full)
        out_stream.write(full)
    out_stream.release()
    # ======================================== End of inferencing ================================================

    # (6) Merge the output video and the input audio.
    logger.info('Merging the output video and the input audio...')
    outfile_path = os.path.join(
        args.output_dir,
        f'{os.path.basename(args.input)[:-4]}_{args.num_ref_images}_{args.num_landmarks_transformer}.mp4'
    )
    command = f'ffmpeg -y -i {args.audio} -i {args.temp_dir}/result.avi -strict -2 -profile:v high {outfile_path}'
    code = subprocess.call(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if code == 0:
        logger.success(f'Output has been saved to: {outfile_path}.')
    else:
        logger.error(f'Failed to merge the output video and the input audio. Exit code: {code}.')


def get_face_masks(original_frames: list, face_crop_results: list):
    """Get face masks for each frame.
    """
    face_masks = []
    contour_idx = list(range(0, 17)) + list(range(17, 27))[::-1]
    face_align = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    for original_frame, (crop_face, face_coordinates) in zip(original_frames, face_crop_results):
        y1, _, x1, _ = face_coordinates
        preds = face_align.get_landmarks_from_image(crop_face)[0].astype(np.int32) + np.array([x1, y1]).astype(np.int32)
        contour_pts = preds[contour_idx]
        mask_img = np.zeros((original_frame.shape[0], original_frame.shape[1], 1), np.uint8)
        cv2.fillConvexPoly(mask_img, contour_pts, 255)
        mask1 = cv2.GaussianBlur(mask_img, (21, 21), 11) / 255
        mask1 = np.tile(np.expand_dims(mask1, axis=2), (1, 1, 3))
        face_masks.append(mask1)
    return face_masks


if __name__ == '__main__':
    main()
