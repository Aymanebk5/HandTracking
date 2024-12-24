import cv2
import mediapipe as mp
from handtrackingmodule import HandDetector
from volumecontrol import VolumeControl

# Initialize Hand Detector and Volume Control
hand_detector = HandDetector()
volume_control = VolumeControl()

# Capture video
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Hand tracking
    results = hand_detector.find_hands(frame)
    hand_landmarks = hand_detector.get_hand_landmarks(frame)

    if hand_landmarks:
        for hand in hand_landmarks:
            # Get thumb and index tip coordinates
            thumb_tip = hand.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
            index_tip = hand.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

            # Calculate the distance between the thumb and index finger
            distance, thumb_pos, index_pos = hand_detector.get_distance(thumb_tip, index_tip, frame)

            # Ensure the distance is within a reasonable range before controlling volume
            if 30 < distance < 200:
                # Map the distance to volume and adjust the volume
                volume_percent = volume_control.set_volume(distance)

                # Display the volume bar and number, passing thumb position and frame
                volume_control.show_volume_bar(distance, thumb_pos, frame)

    # Display the frame with the volume bar and number
    cv2.imshow("Gesture Volume Control", frame)

    # Exit with the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
