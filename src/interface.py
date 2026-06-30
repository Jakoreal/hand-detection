import cv2
import mediapipe as mp
import time
import math

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

PINCH_THRESHOLD = 40  # pixels — tune this based on your camera/distance from screen

# The draggable object: a circle defined by center (x, y) and radius
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
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 0, 0), 2)

    return points


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,  # simpler to start with one hand
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
        # mirror so movement feels natural

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                points = draw_landmarks(frame, hand_landmarks)

                thumb_tip = points[4]
                index_tip = points[8]
                pinch_dist = distance(thumb_tip, index_tip)

                # Midpoint between thumb and index — this is our "pinch cursor"
                pinch_point = (
                    (thumb_tip[0] + index_tip[0]) // 2,
                    (thumb_tip[1] + index_tip[1]) // 2,
                )

                pinching = pinch_dist < PINCH_THRESHOLD

                if pinching and not is_dragging:
                    # Just started pinching — check if we're grabbing the object
                    if distance(pinch_point, obj_pos) < obj_radius:
                        is_dragging = True
                        drag_offset = [
                            obj_pos[0] - pinch_point[0],
                            obj_pos[1] - pinch_point[1],
                        ]

                if pinching and is_dragging:
                    obj_pos[0] = pinch_point[0] + drag_offset[0]
                    obj_pos[1] = pinch_point[1] + drag_offset[1]

                if not pinching:
                    is_dragging = False

                # Visual feedback: pinch cursor color shows pinch state
                cursor_color = (0, 0, 255) if pinching else (0, 255, 255)
                cv2.circle(frame, pinch_point, 8, cursor_color, -1)

        # Draw the draggable object
        obj_color = (0, 200, 0) if is_dragging else (200, 200, 200)
        cv2.circle(frame, tuple(obj_pos), obj_radius, obj_color, -1)

        cv2.imshow("Pinch to Drag", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
