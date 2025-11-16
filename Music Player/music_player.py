import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
import cv2
import mediapipe as mp
import pygame
import logging
from datetime import datetime
from flask import Flask, render_template, jsonify, send_from_directory, Response
import time
import numpy as np

app = Flask(__name__)

# Set up directories
script_dir = os.path.dirname(os.path.abspath(__file__))
music_dir = os.path.join(script_dir, "music")  
lyrics_dir = os.path.join(script_dir, "lyrics")

# Load playlist
def load_playlist():
    if not os.path.exists(music_dir):
        logging.warning(f"{datetime.now()}: 'music' folder not found")
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
    logging.error(f"Pygame init failed: {e}")
    raise

# State
current_index = 0
current_position = 0
is_playing = False
is_camera_active = False
last_gesture_time = 0
gesture_cooldown = 0.7
current_gesture = None
current_volume = 0.5

# CLAP DETECTION
prev_palm_dist = None
last_clap_time = 0
clap_cooldown = 1.8  # Prevent spam

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Camera
def find_usb_camera():
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return 0

camera_index = find_usb_camera()
cap = None

logging.basicConfig(filename='gesture_log.txt', level=logging.INFO)

# === CLAP & GESTURE RECOGNITION ===
def recognize_gesture(landmarks_list):
    global last_gesture_time, current_gesture
    global prev_palm_dist, last_clap_time
    current_time = time.time()

    if current_time - last_gesture_time < gesture_cooldown:
        return None

    # === CLAP: Two hands, palms facing, close fast ===
    if len(landmarks_list) == 2:
        wrist1, wrist2 = landmarks_list[0][0], landmarks_list[1][0]
        palm_center1 = np.array([(landmarks_list[0][9].x + landmarks_list[0][5].x)/2,
                                (landmarks_list[0][9].y + landmarks_list[0][5].y)/2])
        palm_center2 = np.array([(landmarks_list[1][9].x + landmarks_list[1][5].x)/2,
                                (landmarks_list[1][9].y + landmarks_list[1][5].y)/2])
        
        palm_dist = np.linalg.norm(palm_center1 - palm_center2)

        # Check if palms are facing camera (fingers up)
        index1_tip, index2_tip = landmarks_list[0][8], landmarks_list[1][8]
        if (index1_tip.y < landmarks_list[0][6].y and index2_tip.y < landmarks_list[1][6].y):
            if prev_palm_dist is not None:
                dist_drop = prev_palm_dist - palm_dist
                if palm_dist < 0.12 and dist_drop > 0.08 and current_time - last_clap_time > clap_cooldown:
                    last_clap_time = current_time
                    last_gesture_time = current_time
                    current_gesture = "clap"
                    prev_palm_dist = None
                    return "clap"
            prev_palm_dist = palm_dist
        else:
            prev_palm_dist = None
    else:
        prev_palm_dist = None

    # === SINGLE HAND GESTURES (unchanged) ===
    if len(landmarks_list) != 1:
        return None

    wrist = landmarks_list[0][0]
    thumb_tip = landmarks_list[0][4]
    index_tip = landmarks_list[0][8]
    middle_tip = landmarks_list[0][12]
    ring_tip = landmarks_list[0][16]
    pinky_tip = landmarks_list[0][20]

    # Play
    if (index_tip.y < landmarks_list[0][6].y - 0.03 and 
        middle_tip.y < landmarks_list[0][10].y - 0.03 and
        ring_tip.y < landmarks_list[0][14].y - 0.03 and 
        pinky_tip.y < landmarks_list[0][18].y - 0.03):
        last_gesture_time = current_time
        current_gesture = "play"
        return "play"

    # Pause
    if (index_tip.y > landmarks_list[0][6].y + 0.03 and 
        middle_tip.y > landmarks_list[0][10].y + 0.03 and
        ring_tip.y > landmarks_list[0][14].y + 0.03 and 
        pinky_tip.y > landmarks_list[0][18].y + 0.03):
        last_gesture_time = current_time
        current_gesture = "pause"
        return "pause"

    # Previous
    if (index_tip.x < wrist.x - 0.08 and middle_tip.x < wrist.x - 0.08 and
        abs(index_tip.y - middle_tip.y) < 0.03 and
        ring_tip.y > landmarks_list[0][14].y and pinky_tip.y > landmarks_list[0][18].y):
        last_gesture_time = current_time
        current_gesture = "previous"
        return "previous"

    # Next
    if (index_tip.x > wrist.x + 0.08 and middle_tip.x > wrist.x + 0.08 and
        abs(index_tip.y - middle_tip.y) < 0.03 and
        ring_tip.y > landmarks_list[0][14].y and pinky_tip.y > landmarks_list[0][18].y):
        last_gesture_time = current_time
        current_gesture = "next"
        return "next"

    # Volume Up
    if (thumb_tip.y < wrist.y - 0.06 and 
        index_tip.y > landmarks_list[0][6].y + 0.04 and
        middle_tip.y > landmarks_list[0][10].y + 0.04 and
        ring_tip.y > landmarks_list[0][14].y + 0.04 and
        pinky_tip.y > landmarks_list[0][18].y + 0.04):
        last_gesture_time = current_time
        current_gesture = "volume_up"
        return "volume_up"

    # Volume Down
    if (thumb_tip.y > wrist.y + 0.06 and 
        index_tip.y > landmarks_list[0][6].y + 0.04 and
        middle_tip.y > landmarks_list[0][10].y + 0.04 and
        ring_tip.y > landmarks_list[0][14].y + 0.04 and
        pinky_tip.y > landmarks_list[0][18].y + 0.04):
        last_gesture_time = current_time
        current_gesture = "volume_down"
        return "volume_down"

    current_gesture = None
    return None

# VIDEO FEED
def generate_video_feed():
    global is_camera_active, cap, current_gesture
    while is_camera_active:
        if not cap or not cap.isOpened():
            break
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(frame_rgb)
        gesture = None
        all_landmarks = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                all_landmarks.append(hand_landmarks.landmark)
            gesture = recognize_gesture(all_landmarks)
            if gesture:
                logging.info(f"{datetime.now()}: Detected {gesture}")
                handle_gesture(gesture)

        # Show gesture
        if current_gesture:
            text = "CLAP! CAMERA ON" if current_gesture == "clap" else current_gesture.upper()
            color = (0, 255, 255) if current_gesture == "clap" else (0, 255, 0)
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    blank = np.zeros((240, 320, 3), np.uint8)
    _, buffer = cv2.imencode('.jpg', blank)
    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# HANDLE GESTURE
def handle_gesture(gesture):
    global is_camera_active, cap, camera_index
    if gesture == "clap" and not is_camera_active:
        is_camera_active = True
        if not cap:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                logging.info(f"{datetime.now()}: CLAP DETECTED! Camera ON")
            else:
                is_camera_active = False
    elif gesture == "play":
        play_song()
    elif gesture == "pause":
        pygame.mixer.music.pause()
    elif gesture == "next":
        next_song()
    elif gesture == "previous":
        previous_song()
    elif gesture == "volume_up":
        global current_volume
        current_volume = min(1.0, current_volume + 0.02)
        pygame.mixer.music.set_volume(current_volume)
    elif gesture == "volume_down":
        current_volume = max(0.0, current_volume - 0.02)
        pygame.mixer.music.set_volume(current_volume)

# MUSIC
def play_song():
    global current_index, current_position
    if not playlist: return
    path = os.path.join(music_dir, playlist[current_index])
    pygame.mixer.music.load(path)
    pygame.mixer.music.play(start=current_position / 1000.0 if current_position > 0 else 0)

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

# ROUTES
@app.route('/')
def landing(): return render_template('index.html')

@app.route('/player')
def player(): return render_template('music_player_ui.html')

@app.route('/music/<f>')
def serve_music(f): return send_from_directory(music_dir, f)

@app.route('/api/playlist')
def get_playlist(): return jsonify(playlist)

@app.route('/video_feed')
def video_feed(): return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control/<action>', methods=['POST'])
def control(action):
    global is_camera_active, cap
    if action == 'toggle_camera':
        is_camera_active = not is_camera_active
        if not is_camera_active and cap:
            cap.release()
            cap = None
        elif is_camera_active and not cap:
            cap = cv2.VideoCapture(camera_index)
            cap.set(3, 320); cap.set(4, 240)
        return jsonify({'is_camera_active': is_camera_active})
    return jsonify({'status': 'ok'})

@app.route('/api/state')
def get_state():
    return jsonify({
        'current_index': current_index,
        'is_playing': pygame.mixer.music.get_busy(),
        'is_camera_active': is_camera_active,
        'volume': round(current_volume, 2),
        'current_gesture': current_gesture
    })

@app.route('/control/volume/<float:v>', methods=['POST'])
def set_volume(v):
    global current_volume
    if 0 <= v <= 1:
        current_volume = v
        pygame.mixer.music.set_volume(v)
        return jsonify({'volume': v})
    return jsonify({'error': 'invalid'})

if __name__ == "__main__":
    import atexit
    @atexit.register
    def cleanup():
        if cap: cap.release()
        pygame.mixer.quit()
    app.run(debug=False, use_reloader=False)