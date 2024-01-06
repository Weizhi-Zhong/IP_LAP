from dataclasses import dataclass
from typing import Union

import cv2
import mediapipe as mp
import numpy as np
import torch

from draw_landmark import draw_landmarks

from .facemesh import (ALL_LANDMARKS_IDX, CONTENT_LANDMARK_IDX,
                       FACEMESH_CONNECTION, POSE_LANDMARK_IDX)


@dataclass
class MediapipeLandmark:
    idx: int
    x: float
    y: float


def get_facial_landmarks(ori_background_frames, lip_index, ori_sequence_idx):
    mp_face_mesh = mp.solutions.face_mesh

    boxes = []                                                                  # bounding boxes of human face
    lip_dists = []                                                              # lip dists
    # We define the lip dist(openness): distance between the  midpoints of the upper lip and lower lip
    face_crop_results = []
    all_pose_landmarks: list[list[list[Union[int, float]]]] = []
    all_content_landmarks: list[list[list[Union[int, float]]]] = []             # content landmarks include lip and jaw
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                               min_detection_confidence=0.5) as face_mesh:
        # (1) Get bounding boxes and lip dist
        for frame_idx, full_frame in enumerate(ori_background_frames):
            h, w = full_frame.shape[0], full_frame.shape[1]
            results = face_mesh.process(cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                raise NotImplementedError  # not detect face
            face_landmarks = results.multi_face_landmarks[0]

            # Calculate the lip dist
            dx = face_landmarks.landmark[lip_index[0]].x - face_landmarks.landmark[lip_index[1]].x
            dy = face_landmarks.landmark[lip_index[0]].y - face_landmarks.landmark[lip_index[1]].y
            dist = np.linalg.norm((dx, dy))
            lip_dists.append((frame_idx, dist))

            # Get the marginal landmarks to crop face
            x_min, x_max, y_min, y_max = 999, -999, 999, -999
            for idx, landmark in enumerate(face_landmarks.landmark):
                if idx in ALL_LANDMARKS_IDX:
                    x_min = min(x_min, landmark.x)
                    x_max = max(x_max, landmark.x)
                    y_min = min(y_min, landmark.y)
                    y_max = max(y_max, landmark.y)
            # The landmarks coordinates returned by mediapipe range 0~1
            # Plus some pixel to the marginal region
            plus_pixel = 25
            x_min = max(x_min - plus_pixel / w, 0)
            x_max = min(x_max + plus_pixel / w, 1)

            y_min = max(y_min - plus_pixel / h, 0)
            y_max = min(y_max + plus_pixel / h, 1)
            y1, y2, x1, x2 = int(y_min * h), int(y_max * h), int(x_min * w), int(x_max * w)
            boxes.append([y1, y2, x1, x2])
        boxes = np.array(boxes)

        # (2) Croppd face
        face_crop_results = [[image[y1:y2, x1:x2], (y1, y2, x1, x2)]
                             for image, (y1, y2, x1, x2) in zip(ori_background_frames, boxes)]

        # (3) Detect facial landmarks
        for frame_idx, full_frame in enumerate(ori_background_frames):
            h, w = full_frame.shape[0], full_frame.shape[1]
            results = face_mesh.process(cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                raise ValueError("not detect face in some frame!")              # not detect
            face_landmarks = results.multi_face_landmarks[0]

            pose_landmarks, content_landmarks = [], []
            for idx, landmark in enumerate(face_landmarks.landmark):
                if idx in POSE_LANDMARK_IDX:
                    pose_landmarks.append((idx, w * landmark.x, h * landmark.y))
                if idx in CONTENT_LANDMARK_IDX:
                    content_landmarks.append((idx, w * landmark.x, h * landmark.y))

            # Normalize landmarks to 0~1
            y_min, y_max, x_min, x_max = face_crop_results[frame_idx][1]        # bounding boxes
            pose_landmarks = [
                [idx, (x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min)] for idx, x, y in pose_landmarks]
            content_landmarks = [
                [idx, (x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min)] for idx, x, y in content_landmarks]
            all_pose_landmarks.append(pose_landmarks)
            all_content_landmarks.append(content_landmarks)

    all_pose_landmarks = _get_smoothened_landmarks(all_pose_landmarks, windows_T=1)
    all_content_landmarks = _get_smoothened_landmarks(all_content_landmarks, windows_T=1)

    # Arrange the landmark in a certain order, since the landmark index returned by mediapipe is is chaotic.
    for idx in range(len(all_pose_landmarks)):
        all_pose_landmarks[idx] = sorted(all_pose_landmarks[idx],
                                         key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))
        all_content_landmarks[idx] = sorted(all_content_landmarks[idx],
                                            key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))
    return face_crop_results, all_pose_landmarks, all_content_landmarks, lip_dists


def random_select_reference_landmarks(all_pose_landmarks, all_content_landmarks, lip_dists, args):
    # TODO: This is not random select.
    n_ref_lmarks = args.num_landmarks_transformer                               # N_{l} in the paper.
    input_vid_len = args.input_vid_len

    # Randomly select N_l reference landmarks for landmark transformer#
    dists_sorted = sorted(lip_dists, key=lambda x: x[1])
    lip_dist_idx = np.asarray([idx for idx, dist in dists_sorted])  # the frame idxs sorted by lip openness

    ref_lmark_idxs = [lip_dist_idx[int(i)] for i in torch.linspace(0, input_vid_len - 1, steps=n_ref_lmarks)]
    ref_pose = [all_pose_landmarks[reference_idx] for reference_idx in ref_lmark_idxs]
    ref_content = [all_content_landmarks[reference_idx] for reference_idx in ref_lmark_idxs]

    ref_pose_selected = torch.zeros((n_ref_lmarks, 2, 74))                      # 74 landmark
    ref_content_selected = torch.zeros((n_ref_lmarks, 2, 57))                   # 57 landmark
    for idx in range(n_ref_lmarks):
        ref_pose_selected[idx, :, :] = torch.Tensor(ref_pose[idx]).permute(1, 0)[1:, :]                 # (2, 74)
        ref_content_selected[idx, :, :] = torch.Tensor(ref_content[idx]).permute(1, 0)[1:, :]           # (2, 57)
    ref_content_selected = ref_content_selected.unsqueeze(0)                    # (1, N_{l}, 2, 57)
    ref_pose_selected = ref_pose_selected.unsqueeze(0)                          # (1, N_{l}, 2, 74)
    return ref_pose_selected, ref_content_selected, lip_dist_idx


def draw_reference_sketches(face_crop_results, all_pose_landmarks, all_content_landmarks, lip_dist_idx,
                            drawing_spec, args):
    # Unpack config
    ori_sequence_idx = args.ori_sequence_idx
    full_face_landmark_sequence = args.full_face_landmark_sequence
    img_size = args.image_size
    n_reg_img = args.num_ref_images
    input_vid_len = args.input_vid_len

    # select reference images and draw sketches for rendering according to lip openness#
    ref_img_idx = [int(lip_dist_idx[int(i)]) for i in torch.linspace(0, input_vid_len - 1, steps=n_reg_img)]
    ref_imgs = [face_crop_results[idx][0] for idx in ref_img_idx]               # (N, H, W, 3)
    ref_img_pose_landmarks = [all_pose_landmarks[idx] for idx in ref_img_idx]
    ref_img_content_landmarks = [all_content_landmarks[idx] for idx in ref_img_idx]

    ref_img_pose = torch.zeros((n_reg_img, 2, 74))                              # 74 landmark
    ref_img_content = torch.zeros((n_reg_img, 2, 57))                           # 57 landmark
    for idx in range(n_reg_img):
        ref_img_pose[idx, :, :] = torch.Tensor(ref_img_pose_landmarks[idx]).permute(1, 0)[1:, :]        # (2, 74)
        ref_img_content[idx, :, :] = torch.Tensor(ref_img_content_landmarks[idx]).permute(1, 0)[1:, :]  # (2, 57)

    ref_img_full_face_landmarks = torch.cat([ref_img_pose, ref_img_content], dim=2).cpu().numpy()       # (N, 2, 131)
    ref_img_sketches = torch.zeros((ref_img_full_face_landmarks.shape[0], img_size, img_size, 3))

    for frame_idx in range(ref_img_full_face_landmarks.shape[0]):               # N
        full_landmarks = ref_img_full_face_landmarks[frame_idx]                 # (2,131)
        h, w = ref_imgs[frame_idx].shape[0], ref_imgs[frame_idx].shape[1]
        drawn_sketech = np.zeros((int(h * img_size / min(h, w)), int(w * img_size / min(h, w)), 3))
        mediapipe_format_landmarks = [
            MediapipeLandmark(
                ori_sequence_idx[full_face_landmark_sequence[idx]], full_landmarks[0, idx], full_landmarks[1, idx]
            )
            for idx in range(full_landmarks.shape[1])
        ]
        drawn_sketech = draw_landmarks(drawn_sketech, mediapipe_format_landmarks, connections=FACEMESH_CONNECTION,
                                       connection_drawing_spec=drawing_spec)
        drawn_sketech = cv2.resize(drawn_sketech, (img_size, img_size))                         # (128, 128, 3)
        ref_img_sketches[frame_idx] = torch.Tensor(drawn_sketech / 255.0)                       # (N, 128, 128, 3)
    ref_img_sketches = ref_img_sketches.permute(0, 3, 1, 2).unsqueeze(0)                        # (1, N, 3, 128, 128)
    ref_imgs = [cv2.resize(face.copy(), (img_size, img_size)) for face in ref_imgs]
    ref_imgs = torch.Tensor(np.asarray(ref_imgs) / 255.0).unsqueeze(0).permute(0, 1, 4, 2, 3)   # (1, N, 3, H, W)
    return ref_imgs, ref_img_sketches


def _get_smoothened_landmarks(all_landmarks: list, windows_T: int = 1):
    for i in range(len(all_landmarks)):                                         # frame i
        if i + windows_T > len(all_landmarks):
            window = all_landmarks[len(all_landmarks) - windows_T:]
        else:
            window = all_landmarks[i: i + windows_T]

        for j in range(len(all_landmarks[i])):                                  # landmark j
            all_landmarks[i][j][1] = np.mean([frame_landmarks[j][1] for frame_landmarks in window])  # x
            all_landmarks[i][j][2] = np.mean([frame_landmarks[j][2] for frame_landmarks in window])  # y
    return all_landmarks
