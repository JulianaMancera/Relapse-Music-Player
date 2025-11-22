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
import difflib
import webbrowser
from mutagen import File

app = Flask(__name__)

# Set up directories
script_dir = os.path.dirname(os.path.abspath(__file__))
music_dir = os.path.join(script_dir, "music")
lyrics_dir = os.path.join(script_dir, "lyrics")
static_dir = os.path.join(script_dir, "static")
song_durations = {}

# Dynamically load songs
def load_playlist():
    if not os.path.exists(music_dir):
        return []
    songs = [f for f in os.listdir(music_dir) if f.lower().endswith(('.mp3', '.wav', '.ogg', '.flac', '.m4a'))]
    songs.sort(key=str.lower)
    
    # Pre-calculate durations
    global song_durations
    song_durations = {}
    for song in songs:
        try:
            file_path = os.path.join(music_dir, song)
            audio_file = File(file_path)
            if audio_file:
                song_durations[song] = audio_file.info.length
        except:
            song_durations[song] = 0
    
    return songs

playlist = load_playlist()

# Pygame
pygame.mixer.init()
pygame.mixer.music.set_volume(0.5)

current_index = 0
current_position = 0
is_camera_active = False
last_gesture_time = 0
gesture_cooldown = 0.9
current_gesture = None
current_volume = 0.5

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.8
)

cap = None

def log_gesture(gesture):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] GESTURE → {gesture.upper()}")

logging.basicConfig(filename='gesture_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# LYRICS MATCHING 
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

# CAMERA & GESTURE
def find_usb_camera():
    for idx in range(1, 10):
        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        temp = cv2.VideoCapture(idx, backend)
        if temp.isOpened():
            temp.release()
            return idx
        temp.release()
    return None

# Camera
def open_camera():
    global cap
    if cap and cap.isOpened():
        cap.release()
    idx = 0
    for i in range(10):
        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        temp = cv2.VideoCapture(i, backend)
        if temp.isOpened():
            idx = i
            temp.release()
            break
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return True
    return False

def recognize_gesture(landmarks):
    global last_gesture_time, current_gesture
    now = time.time()
    if now - last_gesture_time < gesture_cooldown:
        return None

    tip = lambda i: landmarks[i]
    pip = lambda i: landmarks[i - 2 if i > 4 else i]  # safer PIP access

    # Thumb state (for reference, optional)
    thumb_extended = tip(4).x > tip(3).x + 0.05 if tip(4).x > tip(0).x else tip(4).x < tip(3).x - 0.05

    # Finger extended checks (tip significantly above PIP)
    index_extended   = tip(8).y  < tip(6).y  - 0.05
    middle_extended  = tip(12).y < tip(10).y - 0.05
    ring_extended    = tip(16).y < tip(14).y - 0.05
    pinky_extended   = tip(20).y < tip(18).y - 0.05

    # === 🖕 MIDDLE FINGER DETECTION (THE RICKROLL TRIGGER) ===
    # Only middle finger up, all others clearly DOWN
    if (middle_extended and
        not index_extended and
        not ring_extended and
        not pinky_extended and
        tip(12).y < tip(0).y - 0.1):  # middle finger clearly raised above wrist

        last_gesture_time = now
        current_gesture = "rickroll"
        log_gesture("MIDDLE FINGER → RICKROLL ACTIVATED")
        handle_gesture("rickroll")  # Trigger immediately
        return "rickroll"

    wrist = tip(0)

    # Count how many of the four fingers (index to pinky) have tip clearly above their PIP joint
    extended = sum(1 for i in [8, 12, 16, 20] if tip(i).y < tip(i-2).y - 0.04)

    # Individual finger states (True = clearly extended upward)
    index_extended   = tip(8).y  < tip(6).y  - 0.06
    middle_extended  = tip(12).y < tip(10).y - 0.06
    ring_extended    = tip(16).y < tip(14).y - 0.06
    pinky_extended   = tip(20).y < tip(18).y - 0.06

    # === 1. ONE FINGER UP (only index) → VOLUME UP ===
    if (index_extended and not middle_extended and not ring_extended and not pinky_extended):
        last_gesture_time = now
        current_gesture = "volume_up"
        log_gesture("volume up")
        return "volume_up"

    # === 2. THREE FINGERS UP (index + middle + ring, pinky down) → VOLUME DOWN ===
    if (index_extended and middle_extended and ring_extended and not pinky_extended):
        last_gesture_time = now
        current_gesture = "volume_down"
        log_gesture("volume down")
        return "volume_down"

    # === 3. TWO FINGERS SWIPE RIGHT → NEXT ===
    if extended == 2 and tip(12).x > wrist.x + 0.07: 
        last_gesture_time = now
        current_gesture = "next"
        log_gesture("next")
        return "next"

    # === 4. TWO FINGERS SWIPE LEFT → PREVIOUS ===
    if extended == 2 and tip(12).x < wrist.x - 0.07:
        last_gesture_time = now
        current_gesture = "previous"
        log_gesture("previous")
        return "previous"

    # === 5. OPEN PALM (4+ fingers) → PLAY ===
    if extended >= 4:
        last_gesture_time = now
        current_gesture = "play"
        log_gesture("play")
        return "play"

    # === 6. FIST (0 or 1 finger, but not the single index-up we already caught) → PAUSE ===
    if extended <= 1:
        last_gesture_time = now
        current_gesture = "pause"
        log_gesture("pause")
        return "pause"

    current_gesture = None
    return None

# Video feed
def generate_video_feed():
    global is_camera_active, cap, current_gesture
    while is_camera_active:
        if not cap or not cap.isOpened():
            break
        success, frame = cap.read()
        if not success:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand = max(results.multi_hand_landmarks, key=lambda h: h.landmark[0].z) 
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
                                    mp_drawing.DrawingSpec(color=(255,255,255), thickness=2))
            gesture = recognize_gesture(hand.landmark)
            if gesture:
                handle_gesture(gesture)

        if current_gesture:
            txt = current_gesture.replace("_", " ").upper()
            cv2.putText(frame, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0), 4)
            cv2.putText(frame, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,100), 3)

        _, buf = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

# Gesture handler
def handle_gesture(gesture):
    global current_volume

    if gesture == "rickroll":
        pygame.mixer.music.stop()              
        pygame.mixer.music.set_volume(1.0)        
        import webbrowser
        webbrowser.open("https://youtu.be/dQw4w9WgXcQ?si=K1_RZzgnUsOeezO8")

        log_gesture("RICKROLL EXECUTED — NEVER GONNA GIVE YOU UP")
        return  
    
    if gesture == "play" and not pygame.mixer.music.get_busy():
        play_song()
    elif gesture == "pause":
        pygame.mixer.music.pause()
    elif gesture == "next":
        next_song()
    elif gesture == "previous":
        previous_song()
    elif gesture == "volume_up":
        current_volume = min(1.0, current_volume + 0.06)
        pygame.mixer.music.set_volume(current_volume)
    elif gesture == "volume_down":
        current_volume = max(0.0, current_volume - 0.06)
        pygame.mixer.music.set_volume(current_volume)

def play_song():
    global current_index, current_position
    if not playlist: return
    path = os.path.join(music_dir, playlist[current_index])
    pygame.mixer.music.load(path)
    pygame.mixer.music.set_volume(current_volume)
    pygame.mixer.music.play(start=current_position/1000)
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
    position = 0
    if pygame.mixer.music.get_busy():
        position = pygame.mixer.music.get_pos() / 1000.0
    
    current_song_name = playlist[current_index] if playlist and current_index < len(playlist) else None
    duration = song_durations.get(current_song_name, 0) if current_song_name else 0

    return jsonify({
        'current_index': current_index,
        'is_playing': pygame.mixer.music.get_busy(),
        'is_camera_active': is_camera_active,
        'volume': round(current_volume, 2),
        'position': round(position, 1),
        'duration': round(duration, 1)
    })

@app.route('/control/volume/<float:volume>', methods=['POST'])
def set_volume(volume):
    global current_volume
    if 0.0 <= volume <= 1.0:
        current_volume = volume
        pygame.mixer.music.set_volume(current_volume)
        return jsonify({'status': 'success', 'volume': round(current_volume, 2)})
    return jsonify({'status': 'error'})

@app.route('/control/play_index/<int:index>', methods=['POST'])
def play_index(index):
    global current_index, current_position
    if 0 <= index < len(playlist):
        current_index = index
        current_position = 0
        play_song()
        return jsonify({'status': 'success', 'current_index': current_index})
    return jsonify({'status': 'error'})

@app.route('/control/seek/<float:seconds>', methods=['POST'])
def seek(seconds):
    global current_position
    if not playlist or current_index >= len(playlist):
        return jsonify({'status': 'error'})

    current_position = max(0, seconds * 1000)  

    was_playing = pygame.mixer.music.get_busy()
    pygame.mixer.music.stop()  
    play_song() 

    if was_playing:
        pygame.mixer.music.unpause()

    return jsonify({'status': 'success'})

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