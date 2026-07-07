import cv2
from deepface import DeepFace
from moviepy.editor import VideoFileClip
import librosa
import os


def extract_faces_from_video(video_path, output_folder="frames"):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # Frames por segundo
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Salvar 1 frame por segundo
        if frame_count % frame_rate == 0:
            frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    return output_folder


def analyze_faces(folder_path):
    results = []
    for image_file in sorted(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, image_file)
        try:
            analysis = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
            results.append((image_file, analysis["dominant_emotion"]))
        except Exception as e:
            print(f"Erro ao analisar {image_file}: {e}")
    return results


def extract_audio_from_video(video_path, output_audio_path="audio.wav"):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_audio_path)
    return output_audio_path


def analyze_audio_emotions(audio_path):
    y, sr = librosa.load(audio_path)
    # Aqui você pode usar algum modelo pré-treinado ou técnica para análise de emoção no áudio.
    # Exemplo: pyAudioAnalysis (não implementado no exemplo)
    print("Análise de emoção no áudio não está configurada.")
    return None


if __name__ == "__main__":
    video_path = "seu_video.mp4"
    frames_folder = extract_faces_from_video(video_path)

    print("Analisando emoções faciais...")
    facial_emotions = analyze_faces(frames_folder)
    print(f"Emoções faciais detectadas: {facial_emotions}")

    print("Extraindo e analisando emoções no áudio...")
    audio_path = extract_audio_from_video(video_path)
    audio_emotions = analyze_audio_emotions(audio_path)
    print(f"Emoções de áudio detectadas: {audio_emotions}")
