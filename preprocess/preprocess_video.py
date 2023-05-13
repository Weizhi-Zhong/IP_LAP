import os.path
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, traceback
from tqdm import tqdm
import glob
import argparse
import math
from typing import List, Mapping, Optional, Tuple, Union
import cv2
import dataclasses
import numpy as np
from mediapipe.framework.formats import landmark_pb2

parser = argparse.ArgumentParser()
parser.add_argument('--process_num', type=int, default=6) #number of process in ThreadPool to preprocess the dataset
parser.add_argument('--dataset_video_root', type=str, required=True)
parser.add_argument('--output_sketch_root', type=str, default='./lrs2_sketch128')
parser.add_argument('--output_face_root', type=str, default='./lrs2_face128')
parser.add_argument('--output_landmark_root', type=str, default='./lrs2_landmarks')

args = parser.parse_args()

input_mp4_root = args.dataset_video_root
output_sketch_root = args.output_sketch_root
output_face_root=args.output_face_root
output_landmark_root=args.output_landmark_root



"""MediaPipe solution drawing utils."""
_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_BGR_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)

@dataclasses.dataclass
class DrawingSpec:
    # Color for drawing the annotation. Default to the white color.
    color: Tuple[int, int, int] = WHITE_COLOR
    # Thickness for drawing the annotation. Default to 2 pixels.
    thickness: int = 2
    # Circle radius. Default to 2 pixels.
    circle_radius: int = 2


def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


FACEMESH_LIPS = frozenset([(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                           (17, 314), (314, 405), (405, 321), (321, 375),
                           (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
                           (37, 0), (0, 267),
                           (267, 269), (269, 270), (270, 409), (409, 291),
                           (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
                           (14, 317), (317, 402), (402, 318), (318, 324),
                           (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
                           (82, 13), (13, 312), (312, 311), (311, 310),
                           (310, 415), (415, 308)])

FACEMESH_LEFT_EYE = frozenset([(263, 249), (249, 390), (390, 373), (373, 374),
                               (374, 380), (380, 381), (381, 382), (382, 362),
                               (263, 466), (466, 388), (388, 387), (387, 386),
                               (386, 385), (385, 384), (384, 398), (398, 362)])

FACEMESH_LEFT_IRIS = frozenset([(474, 475), (475, 476), (476, 477),
                                (477, 474)])

FACEMESH_LEFT_EYEBROW = frozenset([(276, 283), (283, 282), (282, 295),
                                   (295, 285), (300, 293), (293, 334),
                                   (334, 296), (296, 336)])

FACEMESH_RIGHT_EYE = frozenset([(33, 7), (7, 163), (163, 144), (144, 145),
                                (145, 153), (153, 154), (154, 155), (155, 133),
                                (33, 246), (246, 161), (161, 160), (160, 159),
                                (159, 158), (158, 157), (157, 173), (173, 133)])

FACEMESH_RIGHT_EYEBROW = frozenset([(46, 53), (53, 52), (52, 65), (65, 55),
                                    (70, 63), (63, 105), (105, 66), (66, 107)])

FACEMESH_RIGHT_IRIS = frozenset([(469, 470), (470, 471), (471, 472),
                                 (472, 469)])

FACEMESH_FACE_OVAL = frozenset([(389, 356), (356, 454),
                                (454, 323), (323, 361), (361, 288), (288, 397),
                                (397, 365), (365, 379), (379, 378), (378, 400),
                                (400, 377), (377, 152), (152, 148), (148, 176),
                                (176, 149), (149, 150), (150, 136), (136, 172),
                                (172, 58), (58, 132), (132, 93), (93, 234),
                                (234, 127), (127, 162)])
# (10, 338), (338, 297), (297, 332), (332, 284),(284, 251), (251, 389) (162, 21), (21, 54),(54, 103), (103, 67), (67, 109), (109, 10)

FACEMESH_NOSE = frozenset([(168, 6), (6, 197), (197, 195), (195, 5), (5, 4), \
                           (4, 45), (45, 220), (220, 115), (115, 48), \
                           (4, 275), (275, 440), (440, 344), (344, 278), ])
FACEMESH_FULL = frozenset().union(*[
    FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL, FACEMESH_NOSE
])
def summarize_landmarks(edge_set):
    landmarks = set()
    for a, b in edge_set:
        landmarks.add(a)
        landmarks.add(b)
    return landmarks

all_landmark_idx = summarize_landmarks(FACEMESH_FULL)
pose_landmark_idx = \
    summarize_landmarks(FACEMESH_NOSE.union(*[FACEMESH_RIGHT_EYEBROW, FACEMESH_RIGHT_EYE, \
                                              FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, ])).union(
        [162, 127, 234, 93, 389, 356, 454, 323])
content_landmark_idx = all_landmark_idx - pose_landmark_idx

def draw_landmarks(
        image: np.ndarray,
        landmark_list: landmark_pb2.NormalizedLandmarkList,
        connections: Optional[List[Tuple[int, int]]] = None,
        landmark_drawing_spec: Union[DrawingSpec,
        Mapping[int, DrawingSpec]] = DrawingSpec(
            color=RED_COLOR),
        connection_drawing_spec: Union[DrawingSpec,
        Mapping[Tuple[int, int],
        DrawingSpec]] = DrawingSpec()):
    """Draws the landmarks and the connections on the image.
  Args:
    image: A three channel BGR image represented as numpy ndarray.
    landmark_list: A normalized landmark list proto message to be annotated on
      the image.
    connections: A list of landmark index tuples that specifies how landmarks to
      be connected in the drawing.
    landmark_drawing_spec: Either a DrawingSpec object or a mapping from
      hand landmarks to the DrawingSpecs that specifies the landmarks' drawing
      settings such as color, line thickness, and circle radius.
      If this argument is explicitly set to None, no landmarks will be drawn.
    connection_drawing_spec: Either a DrawingSpec object or a mapping from
      hand connections to the DrawingSpecs that specifies the
      connections' drawing settings such as color and line thickness.
      If this argument is explicitly set to None, no landmark connections will
      be drawn.

  Raises:
    ValueError: If one of the followings:
      a) If the input image is not three channel BGR.
      b) If any connetions contain invalid landmark index.
  """
    if not landmark_list:
        return
    if image.shape[2] != _BGR_CHANNELS:
        raise ValueError('Input image must contain three channel bgr data.')
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
             landmark.visibility < _VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                 landmark.presence < _PRESENCE_THRESHOLD)):
            continue
        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                       image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    if connections:
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                 f'from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                drawing_spec = connection_drawing_spec[connection] if isinstance(
                    connection_drawing_spec, Mapping) else connection_drawing_spec
                # if start_idx in content_landmark and end_idx in content_landmark:
                cv2.line(image, idx_to_coordinates[start_idx],
                         idx_to_coordinates[end_idx], drawing_spec.color,
                         drawing_spec.thickness)
    # Draws landmark points after finishing the connection lines, which is
    # aesthetically better.

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def process_video_file(mp4_path):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                               min_detection_confidence=0.5) as face_mesh:
        video_stream = cv2.VideoCapture(mp4_path)
        fps = round(video_stream.get(cv2.CAP_PROP_FPS))
        if fps != 25:
            print(mp4_path, ' fps is not 25!!!')
            exit()
        frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            frames.append(frame)

        for frame_idx,full_frame in enumerate(frames):
            h, w = full_frame.shape[0], full_frame.shape[1]
            results = face_mesh.process(cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                continue  # not detect
            face_landmarks=results.multi_face_landmarks[0]

            #(1)normalize landmarks
            x_min=999
            x_max=-999
            y_min=999
            y_max=-999
            pose_landmarks, content_landmarks = [], []
            for idx, landmark in enumerate(face_landmarks.landmark):
                if idx in all_landmark_idx:
                    if landmark.x<x_min:
                        x_min=landmark.x
                    if landmark.x>x_max:
                        x_max=landmark.x

                    if landmark.y<y_min:
                        y_min=landmark.y
                    if landmark.y>y_max:
                        y_max=landmark.y
                ######
                if idx in pose_landmark_idx:
                    pose_landmarks.append((idx,landmark.x,landmark.y))
                if idx in content_landmark_idx:
                    content_landmarks.append((idx,landmark.x,landmark.y))
            ##########plus 5 pixel to size##########
            x_min=max(x_min-5/w,0)
            x_max = min(x_max + 5 / w, 1)
            #
            y_min = max(y_min - 5 / h, 0)
            y_max = min(y_max + 5 / h, 1)
            face_frame=cv2.resize(full_frame[int(y_min*h):int(y_max*h),int(x_min*w):int(x_max*w)],(128,128))

            # update landmarks
            pose_landmarks=[ \
                (idx,(x-x_min)/(x_max-x_min),(y-y_min)/(y_max-y_min)) for idx,x,y in pose_landmarks]
            content_landmarks=[\
                (idx, (x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min)) for idx, x, y in content_landmarks]
            # update drawed landmarks
            for idx,x,y in pose_landmarks + content_landmarks:
                face_landmarks.landmark[idx].x=x
                face_landmarks.landmark[idx].y=y
            #save landmarks
            result_dict={}
            result_dict['pose_landmarks']=pose_landmarks
            result_dict['content_landmarks']=content_landmarks
            out_dir = os.path.join(output_landmark_root, '/'.join(mp4_path[:-4].split('/')[-2:]))
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir,str(frame_idx)),result_dict)

            #save sketch
            h_new=(y_max-y_min)*h
            w_new = (x_max - x_min) * w
            annotated_image = np.zeros((int(h_new * 128 / min(h_new, w_new)), int(w_new * 128 / min(h_new, w_new)), 3))
            draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,  # FACEMESH_CONTOURS  FACEMESH_LIPS
                connections=FACEMESH_FULL,
                connection_drawing_spec=drawing_spec)  # landmark_drawing_spec=None,
            annotated_image = cv2.resize(annotated_image, (128, 128))

            out_dir = os.path.join(output_sketch_root, '/'.join(mp4_path[:-4].split('/')[-2:]))
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, str(frame_idx)+'.png'), annotated_image)

            #save face frame
            out_dir = os.path.join(output_face_root, '/'.join(mp4_path[:-4].split('/')[-2:]))
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, str(frame_idx) + '.png'), face_frame)

def mp_handler(mp4_path):
    try:
        process_video_file(mp4_path)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()


def main():
    print('looking up videos.... ')
    mp4_list = glob.glob(input_mp4_root + '/*/*.mp4')  #example: .../lrs2_video/5536038039829982468/00001.mp4
    print('total videos :', len(mp4_list))

    process_num = args.process_num
    print('process_num: ', process_num)
    p_frames = ThreadPoolExecutor(process_num)
    futures_frames = [p_frames.submit(mp_handler, mp4_path) for mp4_path in mp4_list]
    _ = [r.result() for r in tqdm(as_completed(futures_frames), total=len(futures_frames))]
    print("complete task!")

if __name__ == '__main__':
    main()
