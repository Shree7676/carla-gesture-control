import cv2
import mediapipe as mp
import math
import logging

# Initialize MediaPipe solutions
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# Thresholds for gesture detection
LEFT_TILT_THRESHOLD = -15  # Degrees for left head tilt
RIGHT_TILT_THRESHOLD = 15  # Degrees for right head tilt
MOUTH_OPEN_THRESHOLD = 0.1  # Normalized distance for mouth open

# Helper functions for gesture detection
def get_head_tilt(landmarks, img_w):
    forehead = landmarks[10]  # Forehead center
    chin = landmarks[152]     # Chin center
    forehead_x = forehead.x * img_w
    chin_x = chin.x * img_w
    forehead_y = forehead.y * img_h
    chin_y = chin.y * img_h
    delta_x = forehead_x - chin_x
    delta_y = forehead_y - chin_y
    angle = math.degrees(math.atan2(delta_x, -delta_y))
    return angle

def is_mouth_open(landmarks, img_h):
    top_lip = landmarks[13]
    bottom_lip = landmarks[14]
    forehead = landmarks[10]
    chin = landmarks[152]
    top_lip_y = top_lip.y * img_h
    bottom_lip_y = bottom_lip.y * img_h
    forehead_y = forehead.y * img_h
    chin_y = chin.y * img_h
    mouth_distance = bottom_lip_y - top_lip_y
    face_height = chin_y - forehead_y
    normalized_distance = mouth_distance / face_height
    return normalized_distance

def is_palm_open(hand_landmarks, img_h):
    wrist = hand_landmarks.landmark[0]
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    index_base = hand_landmarks.landmark[5]
    middle_base = hand_landmarks.landmark[9]
    ring_base = hand_landmarks.landmark[13]
    pinky_base = hand_landmarks.landmark[17]
    wrist_y = wrist.y * img_h
    index_tip_y = index_tip.y * img_h
    middle_tip_y = middle_tip.y * img_h
    ring_tip_y = ring_tip.y * img_h
    pinky_tip_y = pinky_tip.y * img_h
    index_base_y = index_base.y * img_h
    middle_base_y = middle_base.y * img_h
    ring_base_y = ring_base.y * img_h
    pinky_base_y = pinky_base.y * img_h
    fingers_extended = (
        index_tip_y < index_base_y and
        middle_tip_y < middle_base_y and
        ring_tip_y < ring_base_y and
        pinky_tip_y < pinky_base_y
    )
    thumb_extended = math.hypot(
        (thumb_tip.x - index_base.x) * img_w,
        (thumb_tip.y - index_base.y) * img_h
    ) > 0.05 * img_h
    return fingers_extended and thumb_extended

def is_index_finger_raised(hand_landmarks, img_h):
    index_tip = hand_landmarks.landmark[8]
    index_base = hand_landmarks.landmark[5]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    thumb_tip = hand_landmarks.landmark[4]
    index_tip_y = index_tip.y * img_h
    index_base_y = index_base.y * img_h
    middle_tip_y = middle_tip.y * img_h
    ring_tip_y = ring_tip.y * img_h
    pinky_tip_y = pinky_tip.y * img_h
    thumb_tip_y = thumb_tip.y * img_h
    index_raised = index_tip_y < index_base_y - 0.05 * img_h
    other_fingers_down = (
        middle_tip_y > index_tip_y + 0.03 * img_h and
        ring_tip_y > index_tip_y + 0.03 * img_h and
        pinky_tip_y > index_tip_y + 0.03 * img_h and
        thumb_tip_y > index_tip_y + 0.03 * img_h
    )
    return index_raised and other_fingers_down

class ExpressionDetector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
        self.hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.logger.error("Could not open webcam")
            self.cap = None
        else:
            self.logger.info("Webcam initialized successfully")
        self.head_tilt_left = False
        self.head_tilt_right = False
        self.mouth_open = False
        self.palm_open = False
        self.index_finger_raised = False

    def update(self):
        if not self.cap or not self.cap.isOpened():
            self.logger.warning("Webcam not available")
            return
        ret, frame = self.cap.read()
        if not ret:
            self.logger.error("Failed to capture frame")
            return
        global img_h, img_w
        img_h, img_w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_mesh.process(frame_rgb)
        hand_results = self.hands.process(frame_rgb)

        # Reset gesture states
        self.head_tilt_left = False
        self.head_tilt_right = False
        self.mouth_open = False
        self.palm_open = False
        self.index_finger_raised = False

        # Process face gestures
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                tilt_angle = get_head_tilt(face_landmarks.landmark, img_w)
                if tilt_angle < LEFT_TILT_THRESHOLD:
                    self.head_tilt_left = True
                elif tilt_angle > RIGHT_TILT_THRESHOLD:
                    self.head_tilt_right = True
                mouth_distance = is_mouth_open(face_landmarks.landmark, img_h)
                if mouth_distance > MOUTH_OPEN_THRESHOLD:
                    self.mouth_open = True

        # Process hand gestures
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                if is_palm_open(hand_landmarks, img_h):
                    self.palm_open = True
                if is_index_finger_raised(hand_landmarks, img_h):
                    self.index_finger_raised = True

        # Draw landmarks for debugging
        self.draw_landmarks(frame, face_results, hand_results)
        # Optional: Display frame for debugging (comment out if not needed)
        cv2.imshow('Gesture Detection', frame)
        cv2.waitKey(1)

    def is_head_tilted_left(self):
        return self.head_tilt_left

    def is_head_tilted_right(self):
        return self.head_tilt_right

    def is_mouth_open(self):
        return self.mouth_open

    def is_palm_open(self):
        return self.palm_open

    def is_index_finger_raised(self):
        return self.index_finger_raised

    def close(self):
        if self.cap:
            self.cap.release()
            self.logger.info("Webcam released")
        cv2.destroyAllWindows()
        self.face_mesh.close()
        self.hands.close()
        self.logger.info("Expression detector closed")

    def draw_landmarks(self, frame, face_results, hand_results):
        # Draw face landmarks
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Key facial landmarks
                forehead = face_landmarks.landmark[10]
                chin = face_landmarks.landmark[152]
                top_lip = face_landmarks.landmark[13]
                bottom_lip = face_landmarks.landmark[14]

                # Draw circles for key points
                cv2.circle(frame, (int(forehead.x * img_w), int(forehead.y * img_h)), 5, (0, 255, 0), -1)  # Green for forehead
                cv2.circle(frame, (int(chin.x * img_w), int(chin.y * img_h)), 5, (0, 0, 255), -1)  # Red for chin
                cv2.circle(frame, (int(top_lip.x * img_w), int(top_lip.y * img_h)), 5, (255, 255, 0), -1)  # Yellow for top lip
                cv2.circle(frame, (int(bottom_lip.x * img_w), int(bottom_lip.y * img_h)), 5, (255, 0, 255), -1)  # Magenta for bottom lip

                # Draw lines between points
                cv2.line(frame, (int(forehead.x * img_w), int(forehead.y * img_h)),
                         (int(chin.x * img_w), int(chin.y * img_h)), (255, 0, 0), 1)  # Blue line forehead to chin
                cv2.line(frame, (int(top_lip.x * img_w), int(top_lip.y * img_h)),
                         (int(bottom_lip.x * img_w), int(bottom_lip.y * img_h)), (0, 255, 255), 1)  # Cyan line top to bottom lip

        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )
