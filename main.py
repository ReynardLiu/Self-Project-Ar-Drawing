import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

canvas = None
selected_color = (0, 0, 255)  # default red
current_tool = "Brush"

palette_boxes = [
    ((20, 20), (70, 70), (0,0,255)),     # Red
    ((80, 20), (130, 70), (0,255,0)),    # Green
    ((140, 20), (190, 70), (255,0,0)),   # Blue
    ((200, 20), (250, 70), (0,255,255)), # Yellow
    ((260, 20), (310, 70), (255,255,255)) # White (eraser color option)
]

prev_x, prev_y = 0, 0

def point_in_box(point, box):
    (x1,y1),(x2,y2),_ = box
    return x1 <= point[0] <= x2 and y1 <= point[1] <= y2

def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame,1)
        h,w,_ = frame.shape

        if canvas is None:
            canvas = np.zeros_like(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):

                lm = hand.landmark
                x = int(lm[8].x * w)
                y = int(lm[8].y * h)
                thumb_x = int(lm[4].x * w)
                thumb_y = int(lm[4].y * h)
                mid_x = int(lm[12].x * w)
                mid_y = int(lm[12].y * h)

                # Detect finger states
                thumb_up = finger_up(lm, 4, 3)
                index_up = finger_up(lm, 8, 6)
                middle_up = finger_up(lm, 12, 10)
                ring_up = finger_up(lm, 16, 14)
                pinky_up = finger_up(lm, 20, 18)

                # === HAND DETECTION BOX + LABEL ===
                xs, ys = [], []
                for i in range(21):
                    xs.append(int(lm[i].x * w))
                    ys.append(int(lm[i].y * h))
                cv2.rectangle(frame, (min(xs)-15, min(ys)-15), (max(xs)+15, max(ys)+15), (255,255,255), 2)

                hand_label = handedness.classification[0].label + " Hand"
                cv2.putText(frame, hand_label, (min(xs)-15, min(ys)-25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                # === TRUE FIST → CLEAR CANVAS ===
                if not thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
                    canvas = np.zeros_like(frame)
                    prev_x, prev_y = 0, 0
                    current_tool = "Brush"
                    continue

                # === COLOR PICK MODE (Thumb Only) ===
                if thumb_up and not index_up and not middle_up:
                    for box in palette_boxes:
                        if point_in_box((thumb_x,thumb_y), box):
                            selected_color = box[2]
                            current_tool = "Brush"
                            (x1,y1),(x2,y2),_ = box
                            cv2.rectangle(frame,(x1-4,y1-4),(x2+4,y2+4),(255,255,255),3)

                # === BRUSH MODE (Index only) ===
                elif index_up and not middle_up:
                    current_tool = "Brush"
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = x, y
                    cv2.line(canvas,(prev_x,prev_y),(x,y),selected_color,7)
                    prev_x, prev_y = x, y

                # === ERASER MODE (Index + Middle) ===
                elif index_up and middle_up:
                    current_tool = "Eraser"
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = x, y
                    cv2.line(canvas,(prev_x,prev_y),(x,y),(0,0,0),40)
                    prev_x, prev_y = x, y

                else:
                    prev_x, prev_y = 0, 0

            frame = cv2.add(frame, canvas)

        # Draw Palette
        for (x1,y1),(x2,y2),color in palette_boxes:
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,-1)

        # Current Color Preview
        cv2.rectangle(frame,(w-80,h-80),(w-20,h-20),selected_color,-1)
        cv2.putText(frame,current_tool,(20,h-20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

        cv2.imshow("AR Paint",frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
