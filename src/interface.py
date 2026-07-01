import cv2
import mediapipe as mp
import time
import math
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]

PINCH_THRESHOLD = 40
BOX_MIN_DIAGONAL = 50

MODES = ["Pinch Drag", "Inversion Box", "Finger Boxes"]
current_mode = 0

# Each face: (left_tip_idx, right_tip_idx, effect)
# Quad is built: left_tip_A → left_tip_B → right_tip_B → right_tip_A
FINGER_FACES = [
    (4, 8, "invert"),  # thumb tip + index tip
    (8, 12, "blur"),  # index tip + middle tip
    (12, 16, "gray"),  # middle tip + ring tip
]

obj_pos = [320, 240]
obj_radius = 40
is_dragging = False
drag_offset = [0, 0]


def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    points = []
    for lm in hand_landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        points.append((x, y))
        # cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    for start, end in HAND_CONNECTIONS:
        # cv2.line(frame, points[start], points[end], (255, 0, 0), 2)
        burger = "Yum"
    return points


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def get_pinch_state(points):
    thumb_tip = points[4]
    index_tip = points[8]
    pinching = distance(thumb_tip, index_tip) < PINCH_THRESHOLD
    midpoint = (
        (thumb_tip[0] + index_tip[0]) // 2,
        (thumb_tip[1] + index_tip[1]) // 2,
    )
    return pinching, midpoint


def draw_inversion_box(frame, p1, p2):
    h, w, _ = frame.shape
    x1 = max(0, min(p1[0], p2[0]))
    y1 = max(0, min(p1[1], p2[1]))
    x2 = min(w, max(p1[0], p2[0]))
    y2 = min(h, max(p1[1], p2[1]))
    if x2 > x1 and y2 > y1:
        roi = frame[y1:y2, x1:x2]
        frame[y1:y2, x1:x2] = cv2.bitwise_not(roi)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.line(frame, p1, p2, (255, 255, 0), 1)


def apply_polygon_effect(frame, pts, effect):
    """Apply a visual effect to an arbitrary polygon region."""
    pts_np = np.array(pts, dtype=np.int32)
    h, w = frame.shape[:2]

    # build a mask the size of the full frame
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_np], 255)

    if effect == "invert":
        effect_frame = cv2.bitwise_not(frame)
    elif effect == "blur":
        effect_frame = cv2.GaussianBlur(frame, (31, 31), 0)
    elif effect == "gray":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        effect_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        return

    # paste effect pixels into frame only where mask is filled
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    frame[:] = np.where(mask_3ch == 255, effect_frame, frame)

    # outline the polygon
    cv2.polylines(frame, [pts_np], isClosed=True, color=(255, 255, 0), thickness=2)


def draw_finger_boxes(frame, left_pts, right_pts):
    for tip_a, tip_b, effect in FINGER_FACES:
        # quad corners going around the shape cleanly
        quad = [
            left_pts[tip_a],
            left_pts[tip_b],
            right_pts[tip_b],
            right_pts[tip_a],
        ]
        apply_polygon_effect(frame, quad, effect)


def draw_hud(frame, mode_name):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(
        frame,
        f"Mode [{current_mode + 1}/{len(MODES)}]: {mode_name}  |  TAB to switch",
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

with HandLandmarker.create_from_options(options) as landmarker:
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        active_pinch_points = []
        left_points = None
        right_points = None

        if result.hand_landmarks:
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                points = draw_landmarks(frame, hand_landmarks)

                # use MediaPipe's handedness label to sort left vs right
                label = result.handedness[i][0].category_name
                if label == "Left":
                    left_points = points
                else:
                    right_points = points

                pinching, pinch_point = get_pinch_state(points)
                cursor_color = (0, 0, 255) if pinching else (0, 255, 255)
                cv2.circle(frame, pinch_point, 8, cursor_color, -1)

                if pinching:
                    active_pinch_points.append(pinch_point)

            # ── MODE 0: Pinch Drag ─────────────────────────────────────────
            if current_mode == 0:
                if len(active_pinch_points) == 1:
                    pinch_point = active_pinch_points[0]
                    if not is_dragging:
                        if distance(pinch_point, obj_pos) < obj_radius:
                            is_dragging = True
                            drag_offset = [
                                obj_pos[0] - pinch_point[0],
                                obj_pos[1] - pinch_point[1],
                            ]
                    if is_dragging:
                        obj_pos[0] = pinch_point[0] + drag_offset[0]
                        obj_pos[1] = pinch_point[1] + drag_offset[1]
                else:
                    is_dragging = False

            # ── MODE 1: Inversion Box ──────────────────────────────────────
            elif current_mode == 1:
                if len(active_pinch_points) == 2:
                    p1, p2 = active_pinch_points
                    if distance(p1, p2) > BOX_MIN_DIAGONAL:
                        draw_inversion_box(frame, p1, p2)

            # ── MODE 2: Finger Boxes ───────────────────────────────────────
            elif current_mode == 2:
                if left_points and right_points:
                    draw_finger_boxes(frame, left_points, right_points)

        else:
            is_dragging = False

        # draggable circle only relevant in mode 0
        if current_mode == 0:
            obj_color = (0, 200, 0) if is_dragging else (200, 200, 200)
            cv2.circle(frame, tuple(obj_pos), obj_radius, obj_color, -1)

        draw_hud(frame, MODES[current_mode])
        cv2.imshow("Hand Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == 9:  # TAB
            current_mode = (current_mode + 1) % len(MODES)
            is_dragging = False

cap.release()
cv2.destroyAllWindows()
