def summarize_landmark(edge_set):  # summarize all ficial landmarks used to construct edge
    landmarks = set()
    for a, b in edge_set:
        landmarks.add(a)
        landmarks.add(b)
    return landmarks


# the following is the connections of landmarks for drawing sketch image
facemesh_lips = frozenset([
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308)
])

facemesh_left_eye = frozenset([
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362),
    (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398), (398, 362)
])

facemesh_left_eyebrow = frozenset([
    (276, 283), (283, 282), (282, 295), (295, 285),
    (300, 293), (293, 334), (334, 296), (296, 336)
])

facemesh_right_eye = frozenset([
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133),
    (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173), (173, 133)
])

facemesh_right_eyebrow = frozenset([
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66), (66, 107)
])

facemesh_face_oval = frozenset([
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397), (397, 365), (365, 379),
    (379, 378), (378, 400), (400, 377), (377, 152), (152, 148), (148, 176), (176, 149), (149, 150),
    (150, 136), (136, 172), (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162)
])

facemesh_nose = frozenset([
    (168, 6), (6, 197), (197, 195), (195, 5), (5, 4), (4, 45), (45, 220),
    (220, 115), (115, 48), (4, 275), (275, 440), (440, 344), (344, 278),
])

FACEMESH_CONNECTION = frozenset().union(*[
    facemesh_lips, facemesh_left_eye, facemesh_left_eyebrow, facemesh_right_eye,
    facemesh_right_eyebrow, facemesh_face_oval, facemesh_nose
])

# pose landmarks are landmarks of the upper-half face(eyes,nose,cheek) that represents the pose information
ALL_LANDMARKS_IDX = summarize_landmark(FACEMESH_CONNECTION)
POSE_LANDMARK_IDX = summarize_landmark(
    facemesh_nose.union(*[facemesh_right_eyebrow, facemesh_right_eye, facemesh_left_eye, facemesh_left_eyebrow, ])
).union([162, 127, 234, 93, 389, 356, 454, 323])

# content_landmark include landmarks of lip and jaw which are inferred from audio
CONTENT_LANDMARK_IDX = ALL_LANDMARKS_IDX - POSE_LANDMARK_IDX
