"""MediaPipe solution drawing utils."""
import math
from typing import List, Mapping, Optional, Tuple, Union
import cv2
import dataclasses
import numpy as np
import tqdm
from mediapipe.framework.formats import landmark_pb2
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
#(10, 338), (338, 297), (297, 332), (332, 284),(284, 251), (251, 389) (162, 21), (21, 54),(54, 103), (103, 67), (67, 109), (109, 10)

FACEMESH_NOSE= frozenset([(168, 6),(6,197),(197,195),(195,5),(5,4),\
                          (4,45),(45,220),(220,115),(115,48),\
                          (4,275),(275,440),(440,344),(344,278),])
FACEMESH_FULL = frozenset().union(*[
    FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL,FACEMESH_NOSE
])
connections=FACEMESH_FULL

def summary_landmark(edge_set):
    landmarks=set()
    for a,b in edge_set:
        landmarks.add(a)
        landmarks.add(b)
    return landmarks
all_landmark_idx=summary_landmark(FACEMESH_FULL)
pose_landmark_idx=\
summary_landmark(FACEMESH_NOSE.union(*[FACEMESH_RIGHT_EYEBROW,FACEMESH_RIGHT_EYE,\
                                       FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW,])).union([162,127,234,93,389,356,454,323])
content_landmark_idx= all_landmark_idx - pose_landmark_idx


def draw_landmarks(
    image: np.ndarray,
    landmark_list: List,
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
  for landmark in landmark_list:
    # if ((landmark.HasField('visibility') and
    #      landmark.visibility < _VISIBILITY_THRESHOLD) or
    #     (landmark.HasField('presence') and
    #      landmark.presence < _PRESENCE_THRESHOLD)):
    #   continue
    idx=landmark.idx
    landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                 image_cols, image_rows)
    if landmark_px:
      idx_to_coordinates[idx] = landmark_px

  if connections:
    num_landmarks = len(landmark_list)
    # Draws the connections if the start and end landmarks are both visible.
    for connection in connections:
      start_idx = connection[0]
      end_idx = connection[1]
      # if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
      #   raise ValueError(f'Landmark index is out of range. Invalid connection '
      #                    f'from landmark #{start_idx} to landmark #{end_idx}.')
      if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
        drawing_spec = connection_drawing_spec[connection] if isinstance(
            connection_drawing_spec, Mapping) else connection_drawing_spec
        # if start_idx in content_landmark and end_idx in content_landmark:
        cv2.line(image, idx_to_coordinates[start_idx],
                 idx_to_coordinates[end_idx], drawing_spec.color,
                 drawing_spec.thickness)
  return image
  # Draws landmark points after finishing the connection lines, which is
  # aesthetically better.
  # if landmark_drawing_spec:
  #   for idx, landmark_px in idx_to_coordinates.items():
  #     drawing_spec = landmark_drawing_spec[idx] if isinstance(
  #         landmark_drawing_spec, Mapping) else landmark_drawing_spec
  #     # White circle border
      # circle_border_radius = max(drawing_spec.circle_radius + 1,
      #                            int(drawing_spec.circle_radius * 1.2))
      # circle_border_radius=circle_border_radius*0.1
      # cv2.circle(image, landmark_px, circle_border_radius, WHITE_COLOR,
      #            drawing_spec.thickness)
      # Fill color into the circle

      # cv2.circle(image, landmark_px, 1,
      #            drawing_spec.color, drawing_spec.thickness)
      # cv2.putText(image,str(idx),landmark_px,cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv2.LINE_AA)
