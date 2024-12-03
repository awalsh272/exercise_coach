import cv2
import numpy as np
import openai
from scipy.io import wavfile
from threading import Thread
import time
import math
from gtts import gTTS
import mediapipe as mp
import subprocess

from utils import generate_response

mp_drawing = mp.solutions.drawing_utils

#from utils import create_deepfake, generate_response, text_to_speech

# Set your OpenAI API key
#openai.api_key = 'YOUR_OPENAI_API_KEY'
REFERENCE_VIDEO = "reference.mp4"


# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# Function to get pose estimation from a frame
def get_pose_estimation(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    return results

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    # a, b, c are (x, y) coordinates
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Function to assess exercise performance
def assess_performance(frames, exercise):
    if not frames:
        return False

    # Analyze the last frame
    frame = frames[-1]
    results = get_pose_estimation(frame)
    
    if not results.pose_landmarks:
        return False

    # Extract landmarks
    landmarks = results.pose_landmarks.landmark

    # Map landmark indices to readable variable names
    # MediaPipe Pose indices can be found in the documentation

    if exercise == 'push-ups':
        # Calculate elbow angles
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

        # Get coordinates
        l_shoulder = [left_shoulder.x, left_shoulder.y]
        l_elbow = [left_elbow.x, left_elbow.y]
        l_wrist = [left_wrist.x, left_wrist.y]

        r_shoulder = [right_shoulder.x, right_shoulder.y]
        r_elbow = [right_elbow.x, right_elbow.y]
        r_wrist = [right_wrist.x, right_wrist.y]

        # Calculate angles
        angle_left = calculate_angle(l_shoulder, l_elbow, l_wrist)
        angle_right = calculate_angle(r_shoulder, r_elbow, r_wrist)

        # Check if elbows are bent
        if angle_left < 90 and angle_right < 90:
            return True

    elif exercise == 'squat jumps':
        # Calculate knee angles
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # Get coordinates
        l_hip = [left_hip.x, left_hip.y]
        l_knee = [left_knee.x, left_knee.y]
        l_ankle = [left_ankle.x, left_ankle.y]

        r_hip = [right_hip.x, right_hip.y]
        r_knee = [right_knee.x, right_knee.y]
        r_ankle = [right_ankle.x, right_ankle.y]

        # Calculate angles
        angle_left = calculate_angle(l_hip, l_knee, l_ankle)
        angle_right = calculate_angle(r_hip, r_knee, r_ankle)

        # Check if knees are bent
        if angle_left < 70 and angle_right < 70:
            return True

    elif exercise == 'burpees':
        # Check for plank position
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        l_shoulder = [left_shoulder.x, left_shoulder.y]
        l_hip = [left_hip.x, left_hip.y]
        l_ankle = [left_ankle.x, left_ankle.y]

        r_shoulder = [right_shoulder.x, right_shoulder.y]
        r_hip = [right_hip.x, right_hip.y]
        r_ankle = [right_ankle.x, right_ankle.y]

        angle_left = calculate_angle(l_shoulder, l_hip, l_ankle)
        angle_right = calculate_angle(r_shoulder, r_hip, r_ankle)

        if angle_left > 160 and angle_right > 160:
            return True

    elif exercise == 'handstand':
        # Check if the person is inverted
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

        if left_wrist.y < nose.y or right_wrist.y < nose.y:
            return True

    elif exercise == 'pistol squat':
        # Check one-leg squat
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        l_hip = [left_hip.x, left_hip.y]
        l_knee = [left_knee.x, left_knee.y]
        l_ankle = [left_ankle.x, left_ankle.y]
        r_ankle = [right_ankle.x, right_ankle.y]

        angle_left = calculate_angle(l_hip, l_knee, l_ankle)

        if angle_left < 70 and r_ankle.y < l_hip.y:
            return True

        # Check the other leg
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

        r_hip = [right_hip.x, right_hip.y]
        r_knee = [right_knee.x, right_knee.y]
        r_ankle = [right_ankle.x, right_ankle.y]
        l_ankle = [left_ankle.x, left_ankle.y]

        angle_right = calculate_angle(r_hip, r_knee, r_ankle)

        if angle_right < 70 and l_ankle.y < r_hip.y:
            return True

    return False

# Function to generate response using LLM
# def generate_response(prompt):
#     response = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=prompt,
#         max_tokens=50
#     )
#     return response.choices[0].text.strip()

# Function to convert text to speech
def text_to_speech(text, filename):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)

# Function to create deepfake video using Wav2Lip
def create_deepfake(video_path, audio_path, output_path):
    command = [
        'python', 'main.py', '--checkpoint_path', 'checkpoints/wav2lip_gan.pth',
        '--face', video_path, '--audio', audio_path, '--outfile', output_path
    ]
    # subprocess.call(command)
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while creating deepfake: {e}")
        raise

# Main function
def main():
    cap = cv2.VideoCapture(0)
    exercises = ['push-ups', 'squat jumps', 'burpees', 'handstand', 'pistol squat']
    current_exercise = 0

    while current_exercise < len(exercises):
        exercise = exercises[current_exercise]
        prompt = f"You are an Austrailian fitness influencer named Kirsty Godso, known as KG. Tell the user to please perform {exercise}."

        # Generate initial prompt and convert to speech
        response_text = generate_response(prompt)
        print("response text")
        print(prompt, response_text)
        audio_filename = 'response.mp3'
        text_to_speech(response_text, audio_filename)

        # Create deepfake video
        video_filename = 'deepfake.mp4'
        create_deepfake(REFERENCE_VIDEO, audio_filename, video_filename)

        print("made deepfake")

        break

        # Play deepfake video
        cap_video = cv2.VideoCapture(video_filename)
        while cap_video.isOpened():
            ret_vid, frame_vid = cap_video.read()
            if not ret_vid:
                break
            cv2.imshow('Deepfake', frame_vid)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cap_video.release()

        # Start assessing the exercise
        start_time = time.time()
        correct_performance = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Assess performance
            correct_performance = assess_performance(results, exercise)

            # Draw landmarks and connections
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                if correct_performance:
                    color = (0, 255, 0)  # Green
                else:
                    color = (0, 0, 255)  # Red

                # Draw pose landmarks with the specified color
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
                )

            cv2.imshow('Exercise', image)

            # Break loop if exercise is performed correctly
            if correct_performance:
                break

            # Break if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        if correct_performance:
            encouragement = generate_response(f"Tell the user they are doing a good job on {exercise}, but haven't done it yet! Keep it up!")
            text_to_speech(encouragement, audio_filename)
            create_deepfake(REFERENCE_VIDEO, audio_filename, video_filename)
            # Play encouragement video
            cap_video = cv2.VideoCapture(video_filename)
            while cap_video.isOpened():
                ret_vid, frame_vid = cap_video.read()
                if not ret_vid:
                    break
                cv2.imshow('Deepfake', frame_vid)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            cap_video.release()
            current_exercise += 1
        else:
            prompt_retry = generate_response(f"Let's start {exercise}! You can do it!")
            text_to_speech(prompt_retry, audio_filename)
            create_deepfake(REFERENCE_VIDEO, audio_filename, video_filename)
            # Play retry video
            cap_video = cv2.VideoCapture(video_filename)
            while cap_video.isOpened():
                ret_vid, frame_vid = cap_video.read()
                if not ret_vid:
                    break
                cv2.imshow('Deepfake', frame_vid)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            cap_video.release()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
