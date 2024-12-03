import subprocess
from gtts import gTTS
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import mediapipe as mp
import cv2


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


# Function to create deepfake video using Wav2Lip
def create_deepfake(video_path, audio_path, output_path):
    command = [
        'python', 'inference.py', '--checkpoint_path', 'checkpoints/wav2lip_gan.pth',
        '--face', video_path, '--audio', audio_path, '--outfile', output_path
    ]
    subprocess.call(command)


# def generate_response(prompt):
#     response = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=prompt,
#         max_tokens=50
#     )
#     return response.choices[0].text.strip()


# def generate_response(prompt: str) -> str:
#     model = AutoModelForCausalLM.from_pretrained(
#         "meta-llama/Llama-2-7b-hf",
#         #load_in_8bit=True,
#         device_map="cpu"
#     )
#     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

#     # Generate text
#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs = model.generate(**inputs, max_new_tokens=50)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_response(prompt):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to convert text to speech
def text_to_speech(text, filename):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)