import cv2
import mediapipe as mp
import random
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Gesture Mapping
gesture_mapping = {
    "rock": "Rock",
    "paper": "Paper",
    "scissors": "Scissors"
}

# Function to classify gestures
def classify_gesture(landmarks):
    # Define finger tip and base landmarks
    finger_tips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
    finger_bases = [landmarks[3], landmarks[6], landmarks[10], landmarks[14], landmarks[18]]

    # Detect gestures based on finger tip positions relative to their base
    extended_fingers = [tip.y < base.y for tip, base in zip(finger_tips, finger_bases)]

    if all(not extended for extended in extended_fingers):  # All fingers folded
        return "rock"
    elif all(extended_fingers):  # All fingers extended
        return "paper"
    elif extended_fingers[1] and extended_fingers[2] and not extended_fingers[0] and not extended_fingers[3] and not extended_fingers[4]:  # Only index and middle extended
        return "scissors"
    else:
        return None

# Function to decide the game outcome
def decide_winner(player_choice, computer_choice):
    if player_choice == computer_choice:
        return "It's a tie!"
    elif (player_choice == "rock" and computer_choice == "scissors") or \
         (player_choice == "paper" and computer_choice == "rock") or \
         (player_choice == "scissors" and computer_choice == "paper"):
        return "You win!"
    else:
        return "Computer wins!"

# Start webcam feed
cap = cv2.VideoCapture(0)
print("Show your hand gesture for Rock, Paper, or Scissors!")

# Initialize game state
computer_choice = None
last_result = ""
winner_message = ""
round_over = False
scan_start_time = 0  # Time when scanning starts

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    if not round_over:  # Game in progress
        player_choice = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get gesture classification
                landmarks = hand_landmarks.landmark
                if scan_start_time == 0:
                    scan_start_time = time.time()  # Start scanning timer

                # Check if the player holds the gesture steady for 1 second
                if time.time() - scan_start_time > 1:
                    player_choice = classify_gesture(landmarks)
                    scan_start_time = 0  # Reset timer

        # Generate computer's choice when player choice is made
        if player_choice:
            computer_choice = random.choice(list(gesture_mapping.keys()))
            last_result = f"You: {gesture_mapping[player_choice]} | Computer: {gesture_mapping[computer_choice]}"
            winner_message = decide_winner(player_choice, computer_choice)
            round_over = True  # Mark the round as completed

    # If the round is over, display results and restart option
    if round_over:
        cv2.putText(frame, last_result, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Winner: {winner_message}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "Press 'r' to Restart or 'q' to Quit", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        # Display scanning instructions
        if scan_start_time != 0:
            cv2.putText(frame, "Hold your gesture steady...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Rock, Paper, Scissors", frame)

    # Handle key press events
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):  # Quit the game
        break
    if key == ord('r'):  # Restart the game
        round_over = False
        last_result = ""
        winner_message = ""
        scan_start_time = 0

# Release resources
cap.release()
cv2.destroyAllWindows()
