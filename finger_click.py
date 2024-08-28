import cv2
import time
import hand_tracking_module
import math

# Camera Parameters
camera_width, camera_height = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, camera_width)
cap.set(4, camera_height)

# FPS (backbone)
past_time = 0

# Create Detector Instance
detector = hand_tracking_module.HandDetector()   

while True:
    success, img = cap.read()

    # Find hands on the img (your camera)
    img = detector.findHands(img)

    # Get position
    landmark_list = detector.findPosition(img, draw=False)
    if len(landmark_list) != 0:

        # Extract coordinates
        x1, y1 = landmark_list[4][1], landmark_list[4][2]
        x2, y2 = landmark_list[8][1], landmark_list[8][2]

        # Create circles around two fingers
        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)

        # Create a line between the selected fingers
        cv2.line(img, (x1, y1), (x2, y2), (255, 100, 100), 3)

        # Calculate center line between selected fingers
        cx, cy = (x1+x2) // 2,( y1 + y2) // 2
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        length_between_fingers = math.hypot(x2 - x1, y2 - y1)
        print(f'length_between_fingers: {length_between_fingers}')

        # Mimicking finger click
        if length_between_fingers < 30:
            print("CLICK")
            # Change color of the circle
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)


    if not success:
        print("Failed to grab frame")
        break

    current_time = time.time()
    fps = 1 / (current_time - past_time)
    past_time = current_time

    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break