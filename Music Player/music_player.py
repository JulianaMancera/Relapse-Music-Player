import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import mediapipe as mp
import pygame
import logging
from datetime import datetime
from flask import Flask, render_template, jsonify, send_from_directory, Response
import threading
import time
import numpy as np
import difflib  # <-- Added for smart lyrics matching

app = Flask(__name__)

# Set up directories
script_dir = os.path.dirname(os.path.abspath(__file__))
music_dir = os.path.join(script_dir, "music")
lyrics_dir = os.path.join(script_dir, "lyrics")
static_dir = os.path.join(script_dir, "static")

# Dynamically load songs
def load_playlist():
    if not os.path.exists(music_dir):
        logging.warning(f"{datetime.now()}: 'music' folder not found at {music_dir}")
        return []
    songs = [f for f in os.listdir(music_dir) if f.lower().endswith(('.mp3', '.wav', '.ogg', '.flac', '.m4a'))]
    songs.sort(key=str.lower)
    logging.info(f"{datetime.now()}: Loaded {len(songs)} songs")
    return songs

playlist = load_playlist()

# Pygame init
try:
    pygame.mixer.init()
    pygame.mixer.music.set_volume(0.5)
except pygame.error as e:
    logging.error(f"Failed to init pygame: {e}")
    raise

current_song = None
current_index = 0
current_position = 0
is_playing = False
is_camera_active = False
last_gesture_time = 0
gesture_cooldown = 0.7
current_gesture = None
current_volume = 0.5

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = None
preferred_cam_idx = None

logging.basicConfig(filename='gesture_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# === SMART LYRICS MATCHING ===
def find_lyrics_file(song_filename):
    if not os.path.exists(lyrics_dir):
        return None

    song_name = os.path.splitext(song_filename)[0].lower().strip()
    lyric_files = [f for f in os.listdir(lyrics_dir) if f.lower().endswith('.txt')]

    if not lyric_files:
        return None

    # Exact match
    for f in lyric_files:
        if os.path.splitext(f)[0].lower() == song_name:
            return f

    # Fuzzy match (80%+ similarity)
    matches = []
    for f in lyric_files:
        lyric_name = os.path.splitext(f)[0].lower()
        ratio = difflib.SequenceMatcher(None, song_name, lyric_name).ratio()
        if ratio >= 0.8:
            matches.append((ratio, f))

    if matches:
        matches.sort(key=lambda x: x[0], reverse=True)
        return matches[0][1]

    return None

# === CAMERA & GESTURES (unchanged) ===
def find_usb_camera():
    for idx in range(1, 10):
        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        temp = cv2.VideoCapture(idx, backend)
        if temp.isOpened():
            temp.release()
            return idx
        temp.release()
    return None

def open_camera():
    global cap, preferred_cam_idx
    if cap and cap.isOpened():
        cap.release()
    usb_idx = find_usb_camera()
    cam_idx = usb_idx if usb_idx is not None else 0
    backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
    cap = cv2.VideoCapture(cam_idx, backend)
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 30)
    preferred_cam_idx = cam_idx
    return True

def recognize_gesture(landmarks):
    global last_gesture_time, current_gesture
    current_time = time.time()
    if current_time - last_gesture_time < gesture_cooldown:
        return None

    wrist = landmarks[0]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    thumb_tip = landmarks[4]

    # Play, Pause, Next, Prev, Volume Up/Down (same as before)
    if all(l.y < landmarks[i].y - 0.03 for i, l in [(6, landmarks[8]), (10, landmarks[12]), (14, landmarks[16]), (18, landmarks[20])]):
        last_gesture_time = current_time
        current_gesture = "play"
        return "play"
    if all(l.y > landmarks[i].y + 0.03 for i, l in [(6, landmarks[8]), (10, landmarks[12]), (14, landmarks[16]), (18, landmarks[20])]):
        last_gesture_time = current_time
        current_gesture = "pause"
        return "pause"
    if index_tip.x < wrist.x - 0.08 and middle_tip.x < wrist.x - 0.08:
        last_gesture_time = current_time
        current_gesture = "previous"
        return "previous"
    if index_tip.x > wrist.x + 0.08 and middle_tip.x > wrist.x + 0.08:
        last_gesture_time = current_time
        current_gesture = "next"
        return "next"
    if thumb_tip.y < wrist.y - 0.06 and all(l.y > landmarks[i].y + 0.04 for i, l in [(6, landmarks[8]), (10, landmarks[12]), (14, landmarks[16]), (18, landmarks[20])]):
        last_gesture_time = current_time
        current_gesture = "volume_up"
        return "volume_up"
    if thumb_tip.y > wrist.y + 0.06 and all(l.y > landmarks[i].y + 0.04 for i, l in [(6, landmarks[8]), (10, landmarks[12]), (14, landmarks[16]), (18, landmarks[20])]):
        last_gesture_time = current_time
        current_gesture = "volume_down"
        return "volume_down"

    current_gesture = None
    return None

def generate_video_feed():
    global is_camera_active, cap, current_gesture
    while is_camera_active:
        if not cap or not cap.isOpened():
            break
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        gesture = None
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                gesture = recognize_gesture(hand.landmark)
                if gesture:
                    handle_gesture(gesture)
        if current_gesture:
            cv2.putText(frame, f"Gesture: {current_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    blank = np.zeros((240, 320, 3), np.uint8)
    _, buf = cv2.imencode('.jpg', blank)
    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

def handle_gesture(gesture):
    global current_volume
    if gesture == "play" and not pygame.mixer.music.get_busy():
        play_song()
    elif gesture == "pause":
        pygame.mixer.music.pause()
    elif gesture == "next":
        next_song()
    elif gesture == "previous":
        previous_song()
    elif gesture == "volume_up":
        current_volume = min(1.0, current_volume + 0.02)
        pygame.mixer.music.set_volume(current_volume)
    elif gesture == "volume_down":
        current_volume = max(0.0, current_volume - 0.02)
        pygame.mixer.music.set_volume(current_volume)

def play_song():
    global current_song, current_position
    if not playlist or current_index >= len(playlist):
        return
    current_song = os.path.join(music_dir, playlist[current_index])
    pygame.mixer.music.load(current_song)
    pygame.mixer.music.set_volume(current_volume)
    pygame.mixer.music.play(start=current_position / 1000.0 if current_position > 0 else 0)
    current_position = 0

def next_song():
    global current_index, current_position
    current_position = 0
    current_index = (current_index + 1) % len(playlist)
    play_song()

def previous_song():
    global current_index, current_position
    current_position = 0
    current_index = (current_index - 1) % len(playlist)
    play_song()

# === ROUTES ===
@app.route('/')
def landing():
    return render_template('index.html')

@app.route('/player')
def player():
    return render_template('music_player_ui.html')

@app.route('/music/<filename>')
def serve_music(filename):
    return send_from_directory(music_dir, filename)

@app.route('/lyrics/<path:filename>')
def serve_lyrics(filename):
    lyrics_file = find_lyrics_file(filename)
    if not lyrics_file:
        return "No lyrics found.", 404
    return send_from_directory(lyrics_dir, lyrics_file)

@app.route('/api/playlist')
def get_playlist():
    return jsonify(playlist)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control/<action>', methods=['POST'])
def control(action):
    global is_playing, is_camera_active
    if action == 'play':
        play_song()
        is_playing = True
    elif action == 'pause':
        pygame.mixer.music.pause()
        is_playing = False
    elif action == 'next':
        next_song()
        is_playing = True
    elif action == 'previous':
        previous_song()
        is_playing = True
    elif action == 'toggle_camera':
        is_camera_active = not is_camera_active
        if is_camera_active and not open_camera():
            is_camera_active = False
        return jsonify({'status': 'success', 'is_camera_active': is_camera_active})
    return jsonify({'status': 'success'})

@app.route('/api/state')
def get_state():
    return jsonify({
        'current_index': current_index,
        'is_playing': pygame.mixer.music.get_busy(),
        'is_camera_active': is_camera_active,
        'volume': round(current_volume, 2)
    })

@app.route('/control/volume/<float:volume>', methods=['POST'])
def set_volume(volume):
    global current_volume
    if 0.0 <= volume <= 1.0:
        current_volume = volume
        pygame.mixer.music.set_volume(current_volume)
        return jsonify({'status': 'success', 'volume': round(current_volume, 2)})
    return jsonify({'status': 'error'})

if __name__ == "__main__":
    import atexit
    @atexit.register
    def cleanup():
        global is_camera_active, cap
        is_camera_active = False
        if cap and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
    app.run(debug=False, use_reloader=False)