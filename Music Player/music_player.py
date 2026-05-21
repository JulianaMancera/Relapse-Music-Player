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
import difflib
import webbrowser
from mutagen import File
from pathlib import Path

# Try to import ML-based ASL recognizer
try:
    from asl_recognizer import ASLRecognizer
    ML_ASL_AVAILABLE = True
except (ImportError, FileNotFoundError):
    ML_ASL_AVAILABLE = False
    print("⚠ ML-based ASL recognizer not available. Train model first: python train_asl_model.py")

# DIRECTORIES
script_dir = os.path.dirname(os.path.abspath(__file__))
music_dir = os.path.join(script_dir, "music")
lyrics_dir = os.path.join(script_dir, "lyrics")
templates_dir = os.path.join(script_dir, "templates")
static_dir = os.path.join(script_dir, "static")

# Flask app with explicit folder paths
app = Flask(__name__, 
    template_folder=templates_dir,
    static_folder=static_dir,
    static_url_path='/static'
)
song_durations = {}
song_metadata = {}
rickroll_triggered = False

# PLAYLIST
def load_playlist():
    if not os.path.exists(music_dir):
        return []
    songs = [f for f in os.listdir(music_dir) if f.lower().endswith(('.mp3', '.wav', '.ogg', '.flac', '.m4a'))]
    songs.sort(key=str.lower)
    global song_durations, song_metadata
    song_durations = {}
    song_metadata = {}
    for song in songs:
        file_path = os.path.join(music_dir, song)
        try:
            audio_file = File(file_path)
            song_durations[song] = audio_file.info.length if audio_file else 0
        except:
            song_durations[song] = 0
        try:
            audio_easy = File(file_path, easy=True)
            tags = audio_easy.tags if audio_easy else None
            if tags:
                def tag_val(key, fallback=''):
                    v = tags.get(key, [fallback])
                    return str(v[0]) if v else fallback
                song_metadata[song] = {
                    'title': tag_val('title', os.path.splitext(song)[0]),
                    'artist': tag_val('artist'),
                    'album': tag_val('album'),
                }
            else:
                song_metadata[song] = {'title': os.path.splitext(song)[0], 'artist': '', 'album': ''}
        except:
            song_metadata[song] = {'title': os.path.splitext(song)[0], 'artist': '', 'album': ''}
    return songs

playlist = load_playlist()

# PYGAME 
pygame.mixer.init()
pygame.mixer.music.set_volume(0.5)

# GLOBALS
current_index = 0
current_position = 0
is_camera_active = False
last_gesture_time = 0
gesture_cooldown = 1.0
current_gesture = None
current_volume = 0.5
cap = None

# ASL RECOGNIZER (ML-based)
asl_recognizer = None
if ML_ASL_AVAILABLE:
    try:
        asl_recognizer = ASLRecognizer()
        print("✓ ML-based ASL Recognizer initialized successfully")
    except Exception as e:
        print(f"⚠ Failed to initialize ASL recognizer: {e}")
        ML_ASL_AVAILABLE = False

# SEARCH
search_buffer = ""
last_search_time = 0
last_letter_time = 0
letter_cooldown = 0.7  # Reduced for faster input, adjust as needed
SEARCH_TIMEOUT = 3.0  # Slightly reduced for quicker search

# MEDIAPIPE 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,  # Lowered slightly for better detection
    min_tracking_confidence=0.7
)

def log_gesture(gesture):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] GESTURE → {gesture.upper()}")

logging.basicConfig(filename='gesture_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# CAMERA
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

def close_camera():
    global cap
    if cap and cap.isOpened():
        cap.release()
        cap = None

# IMPROVED ASL LETTER RECOGNITION
def recognize_asl_letter(landmarks):
    # Helper functions
    def get_landmark(id):
        return landmarks.landmark[id]

    def is_finger_extended(finger_tip_id, finger_pip_id, threshold=0.05):
        tip_y = get_landmark(finger_tip_id).y
        pip_y = get_landmark(finger_pip_id).y
        return tip_y < pip_y - threshold  # Assuming fingers point up, lower y is higher

    def is_finger_curled(finger_tip_id, finger_pip_id, threshold=0.05):
        tip_y = get_landmark(finger_tip_id).y
        pip_y = get_landmark(finger_pip_id).y
        return tip_y > pip_y + threshold

    def dist(id1, id2):
        p1 = get_landmark(id1)
        p2 = get_landmark(id2)
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)  # Use 3D dist for better accuracy

    # Finger IDs
    thumb_tip, thumb_ip = 4, 3
    index_tip, index_dip, index_pip, index_mcp = 8, 7, 6, 5
    middle_tip, middle_dip, middle_pip, middle_mcp = 12, 11, 10, 9
    ring_tip, ring_dip, ring_pip, ring_mcp = 16, 15, 14, 13
    pinky_tip, pinky_dip, pinky_pip, pinky_mcp = 20, 19, 18, 17
    wrist = 0

    # Finger state helpers with adjusted thresholds
    thumb_extended = dist(thumb_tip, wrist) > dist (thumb_ip, wrist) * 1.1  # Approximate extension
    index_extended = is_finger_extended(index_tip, index_pip)
    middle_extended = is_finger_extended(middle_tip, middle_pip)
    ring_extended = is_finger_extended(ring_tip, ring_pip)
    pinky_extended = is_finger_extended(pinky_tip, pinky_pip)

    index_curled = is_finger_curled(index_tip, index_pip)
    middle_curled = is_finger_curled(middle_tip, middle_pip)
    ring_curled = is_finger_curled(ring_tip, ring_pip)
    pinky_curled = is_finger_curled(pinky_tip, pinky_pip)

    # A: Fist, thumb beside fingers
    if index_curled and middle_curled and ring_curled and pinky_curled and get_landmark(thumb_tip).x > get_landmark(index_mcp).x and not thumb_extended:
        return 'A'

    # S: Fist, thumb over fingers
    if index_curled and middle_curled and ring_curled and pinky_curled and get_landmark(thumb_tip).y < get_landmark(index_mcp).y:
        return 'S'

    # T: Fist, thumb between index and middle
    if index_curled and middle_curled and ring_curled and pinky_curled and dist(thumb_tip, index_pip) < 0.05 and dist(thumb_tip, middle_pip) < 0.05:
        return 'T'

    # E: Fist, fingers over thumb
    if index_curled and middle_curled and ring_curled and pinky_curled and get_landmark(thumb_tip).y > get_landmark(index_tip).y:
        return 'E'

    # M: Thumb under index, middle, ring
    if index_curled and middle_curled and ring_curled and pinky_extended and dist(thumb_tip, index_dip) < 0.05 and dist(thumb_tip, middle_dip) < 0.05 and dist(thumb_tip, ring_dip) < 0.05:
        return 'M'

    # N: Thumb under index, middle
    if index_curled and middle_curled and ring_extended and pinky_extended and dist(thumb_tip, index_dip) < 0.05 and dist(thumb_tip, middle_dip) < 0.05:
        return 'N'

    # O: Fingers curled to touch thumb like O
    if dist(thumb_tip, index_tip) < 0.05 and middle_curled and ring_curled and pinky_curled:
        return 'O'

    # B: All fingers extended, thumb folded in
    if index_extended and middle_extended and ring_extended and pinky_extended and not thumb_extended and abs(get_landmark(index_tip).x - get_landmark(pinky_tip).x) < 0.15:
        return 'B'

    # F: Thumb and index form circle, others extended
    if dist(thumb_tip, index_tip) < 0.05 and middle_extended and ring_extended and pinky_extended and is_finger_curled(index_tip, index_dip, 0.02):  # Index slightly bent
        return 'F'

    # D: Index extended, thumb touches middle, others curled
    if index_extended and middle_curled and ring_curled and pinky_curled and dist(thumb_tip, middle_tip) < 0.05:
        return 'D'

    # G: Index and thumb extended horizontal, others curled
    if index_extended and thumb_extended and middle_curled and ring_curled and pinky_curled and abs(get_landmark(thumb_tip).y - get_landmark(index_tip).y) < 0.1:
        return 'G'

    # H: Index and middle extended, parallel, others curled
    if index_extended and middle_extended and ring_curled and pinky_curled and not thumb_extended and abs(get_landmark(index_tip).x - get_landmark(middle_tip).x) < 0.05:
        return 'H'

    # I: Pinky extended, others curled, thumb over
    if pinky_extended and index_curled and middle_curled and ring_curled and not thumb_extended:
        return 'I'

    # K: Index and middle extended, thumb on middle pip
    if index_extended and middle_extended and ring_curled and pinky_curled and dist(thumb_tip, middle_pip) < 0.05:
        return 'K'

    # L: Index and thumb extended forming L
    if index_extended and thumb_extended and middle_curled and ring_curled and pinky_curled and get_landmark(thumb_tip).x < get_landmark(index_tip).x - 0.1:
        return 'L'

    # R: Index and middle extended, crossed
    if index_extended and middle_extended and ring_curled and pinky_curled and not thumb_extended and dist(index_tip, middle_tip) < 0.05 and get_landmark(index_tip).x > get_landmark(middle_tip).x:
        return 'R'

    # U: Index and middle extended, together
    if index_extended and middle_extended and ring_curled and pinky_curled and not thumb_extended and dist(index_tip, middle_tip) < 0.05:
        return 'U'

    # V: Index and middle extended, spread
    if index_extended and middle_extended and ring_curled and pinky_curled and not thumb_extended and dist(index_tip, middle_tip) > 0.1:
        return 'V'

    # W: Index, middle, ring extended, spread
    if index_extended and middle_extended and ring_extended and pinky_curled and not thumb_extended and dist(index_tip, ring_tip) > 0.15:
        return 'W'

    # X: Index bent (hook), others curled
    if is_finger_curled(index_tip, index_dip, 0.0) and not index_extended and middle_curled and ring_curled and pinky_curled and thumb_extended:
        return 'X'

    # Y: Thumb and pinky extended, others curled
    if thumb_extended and pinky_extended and index_curled and middle_curled and ring_curled:
        return 'Y'

    # C: Curved like C, thumb and fingers form arc
    if thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended and dist(thumb_tip, index_tip) > 0.15 and all(is_finger_curled(fid, fid-2, 0.02) for fid in [8,12,16,20]):  # Slight curl
        return 'C'

    # P, Q are often dynamic or oriented down, approximate as K, G but check orientation (higher y for tips)
    if index_extended and middle_extended and ring_curled and pinky_curled and dist(thumb_tip, middle_pip) < 0.05 and get_landmark(index_tip).y > get_landmark(wrist).y:  # Pointing down
        return 'P'

    # Z: Similar to index extended, but since dynamic, approximate as 'I' with motion, but static: index extended, others curled
    if index_extended and middle_curled and ring_curled and pinky_curled and thumb_extended:  # Approximate
        return 'Z'

    return None

# GESTURE RECOGNITION (kept similar, minor adjustments)
def recognize_gesture(landmarks):
    global last_gesture_time, current_gesture, rickroll_triggered
    now = time.time()
    if now - last_gesture_time < gesture_cooldown:
        return None

    tip = lambda i: landmarks.landmark[i]

    # Finger states with adjusted thresholds
    index_extended   = tip(8).y  < tip(6).y  - 0.05
    middle_extended  = tip(12).y < tip(10).y - 0.05
    ring_extended    = tip(16).y < tip(14).y - 0.05
    pinky_extended   = tip(20).y < tip(18).y - 0.05

    if (middle_extended and not index_extended and not ring_extended and not pinky_extended
        and tip(12).y < tip(0).y - 0.1 and not rickroll_triggered):

        rickroll_triggered = True
        last_gesture_time = now
        current_gesture = "rickroll"
        log_gesture("MIDDLE FINGER → RICKROLL")
        return "rickroll"

    # HAND GESTURES
    wrist = tip(0)
    extended_count = sum(1 for i in [8, 12, 16, 20] if tip(i).y < tip(i-2).y - 0.04)

    index_ext  = tip(8).y  < tip(6).y  - 0.06
    middle_ext = tip(12).y < tip(10).y - 0.06
    ring_ext   = tip(16).y < tip(14).y - 0.06
    pinky_ext  = tip(20).y < tip(18).y - 0.06

    if index_ext and not middle_ext and not ring_ext and not pinky_ext:
        last_gesture_time = now
        current_gesture = "volume_up"
        return "volume_up"

    if index_ext and middle_ext and ring_ext and not pinky_ext:
        last_gesture_time = now
        current_gesture = "volume_down"
        return "volume_down"

    if extended_count == 2 and tip(12).x > wrist.x + 0.07:
        last_gesture_time = now
        current_gesture = "next"
        return "next"

    if extended_count == 2 and tip(12).x < wrist.x - 0.07:
        last_gesture_time = now
        current_gesture = "previous"
        return "previous"

    if extended_count >= 4:
        last_gesture_time = now
        current_gesture = "play"
        return "play"

    if extended_count <= 1:
        last_gesture_time = now
        current_gesture = "pause"
        return "pause"

    current_gesture = None
    return None

def perform_search(buffer):
    if not buffer:
        return
    query = buffer.lower()
    best_ratio = 0
    best_index = None
    for i, song in enumerate(playlist):
        meta = song_metadata.get(song, {})
        candidates = [
            meta.get('title', ''),
            meta.get('artist', ''),
            meta.get('album', ''),
            os.path.splitext(song)[0],
        ]
        for candidate in candidates:
            if not candidate:
                continue
            ratio = difflib.SequenceMatcher(None, query, candidate.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_index = i
    if best_index is not None and best_ratio >= 0.6:
        global current_index, current_position
        current_index = best_index
        current_position = 0
        play_song()
        log_gesture(f"SEARCHED FOR '{buffer}' → PLAYING {playlist[best_index]}")
    else:
        log_gesture(f"SEARCHED FOR '{buffer}' → NO MATCH FOUND")

def generate_video_feed():
    global is_camera_active, cap, current_gesture, rickroll_triggered, search_buffer, last_search_time, last_letter_time

    while True:
        if rickroll_triggered and not is_camera_active:
            frame = np.zeros((240, 320, 3), np.uint8)
            cv2.putText(frame, "RICKROLLED!", (20, 90),  cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 4)
            cv2.putText(frame, "Never Gonna Give You Up", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, "Click to re-enable camera", (25, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            _, buf = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue

        if not is_camera_active:
            time.sleep(0.1)
            continue

        if not cap or not cap.isOpened():
            if not open_camera():
                time.sleep(0.1)
                continue

        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture = None
        asl_letter = None
        
        if results.multi_hand_landmarks:
            hand = max(results.multi_hand_landmarks, key=lambda h: h.landmark[0].z)  # Closest hand
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
                                      mp_drawing.DrawingSpec(color=(255,255,255), thickness=2))

            # ML-based ASL Recognition
            if ML_ASL_AVAILABLE and asl_recognizer:
                try:
                    # Extract hand region from frame for better accuracy
                    h, w = frame.shape[:2]
                    
                    # Get hand bounding box from landmarks
                    x_coords = [lm.x for lm in hand.landmark]
                    y_coords = [lm.y for lm in hand.landmark]
                    
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    # Add padding
                    padding = 0.2
                    x_min = max(0, int((x_min - padding) * w))
                    x_max = min(w, int((x_max + padding) * w))
                    y_min = max(0, int((y_min - padding) * h))
                    y_max = min(h, int((y_max + padding) * h))
                    
                    # Extract hand region
                    hand_region = frame[y_min:y_max, x_min:x_max]
                    
                    # Recognize using ML model
                    asl_letter = asl_recognizer.recognize(hand_region)
                
                except Exception as e:
                    pass  # Silently handle errors and fallback to old method
            
            # Fallback to rule-based gesture recognition
            gesture = recognize_gesture(hand)

            now = time.time()

            if asl_letter and not gesture:  # Only append if no control gesture to avoid conflicts
                if now - last_letter_time > letter_cooldown:
                    search_buffer += asl_letter
                    last_search_time = now
                    last_letter_time = now
                    log_gesture(f"ASL LETTER {asl_letter} ADDED TO SEARCH")
                    print(f"Detected ASL letter: {asl_letter} (added to search)")

            if gesture:
                handle_gesture(gesture)

            if asl_letter:
                cv2.putText(frame, f"ASL: {asl_letter}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
                if ML_ASL_AVAILABLE:
                    cv2.putText(frame, "[ML]", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

        if current_gesture:
            txt = current_gesture.replace("_", " ").upper()
            cv2.putText(frame, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0), 4)
            cv2.putText(frame, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,100), 3)

        # Display search buffer
        if search_buffer:
            cv2.putText(frame, f"Search: {search_buffer}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        # Check for search timeout
        now = time.time()
        if search_buffer and now - last_search_time > SEARCH_TIMEOUT:
            perform_search(search_buffer)
            search_buffer = ""

        _, buf = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

def handle_gesture(gesture):
    global current_volume
    if gesture == "rickroll":
        pygame.mixer.music.stop()
        pygame.mixer.music.set_volume(1.0)
        webbrowser.open("https://youtu.be/dQw4w9WgXcQ")
        log_gesture("RICKROLL EXECUTED")
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

# FLASK ROUTES
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

@app.route('/api/metadata')
def get_metadata():
    return jsonify(song_metadata)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control/toggle_camera', methods=['POST'])
def control_toggle_camera():
    global is_camera_active
    is_camera_active = not is_camera_active
    if is_camera_active:
        open_camera()
    else:
        close_camera()
    return jsonify({'status': 'success', 'is_camera_active': is_camera_active})

@app.route('/control/play', methods=['POST'])
def control_play():
    if playlist and not pygame.mixer.music.get_busy():
        pygame.mixer.music.unpause()
        if not pygame.mixer.music.get_busy():
            play_song()
    return jsonify({'status': 'success'})

@app.route('/control/pause', methods=['POST'])
def control_pause():
    pygame.mixer.music.pause()
    return jsonify({'status': 'success'})

@app.route('/control/next', methods=['POST'])
def control_next():
    next_song()
    return jsonify({'status': 'success'})

@app.route('/control/previous', methods=['POST'])
def control_previous():
    previous_song()
    return jsonify({'status': 'success'})

@app.route('/control/volume/<float:level>', methods=['POST'])
def control_volume(level):
    global current_volume
    current_volume = max(0.0, min(1.0, level))
    pygame.mixer.music.set_volume(current_volume)
    return jsonify({'status': 'success', 'volume': round(current_volume, 2)})

@app.route('/control/seek/<float:position>', methods=['POST'])
def control_seek(position):
    global current_position
    if playlist and pygame.mixer.music.get_busy():
        pygame.mixer.music.play(start=position)
    current_position = 0
    return jsonify({'status': 'success'})

@app.route('/control/play_index/<int:index>', methods=['POST'])
def control_play_index(index):
    global current_index, current_position
    if playlist and 0 <= index < len(playlist):
        current_index = index
        current_position = 0
        play_song()
    return jsonify({'status': 'success'})

@app.route('/api/cameras')
def api_cameras():
    available = []
    for i in range(5):
        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        temp = cv2.VideoCapture(i, backend)
        if temp.isOpened():
            available.append(i)
            temp.release()
    return jsonify({'available': available, 'current': 0 if cap and cap.isOpened() else -1})

@app.route('/control/camera/<int:index>', methods=['POST'])
def control_camera(index):
    global cap
    if cap and cap.isOpened():
        cap.release()
    backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return jsonify({'status': 'success', 'camera': index})
    return jsonify({'status': 'error', 'message': f'Camera {index} not available'}), 400

@app.route('/api/state')
def get_state():
    position = pygame.mixer.music.get_pos() / 1000.0 if pygame.mixer.music.get_busy() else 0
    current_song = playlist[current_index] if playlist else None
    duration = song_durations.get(current_song, 0) if current_song else 0

    return jsonify({
        'current_index': current_index,
        'is_playing': pygame.mixer.music.get_busy(),
        'is_camera_active': is_camera_active,
        'volume': round(current_volume, 2),
        'position': round(position, 1),
        'duration': round(duration, 1),
        'search_buffer': search_buffer
    })

@app.route('/reset_rickroll')
def reset_rickroll():
    global rickroll_triggered
    rickroll_triggered = False
    return "<h1>RICKROLL RE-ARMED</h1><p>Ready for next victim</p>"

# LYRICS FINDER
def find_lyrics_file(song_filename):
    if not os.path.exists(lyrics_dir):
        return None
    song_name = os.path.splitext(song_filename)[0].lower().strip()
    lyric_files = [f for f in os.listdir(lyrics_dir) if f.lower().endswith('.txt')]
    if not lyric_files:
        return None
    for f in lyric_files:
        if os.path.splitext(f)[0].lower() == song_name:
            return f
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

if __name__ == "__main__":
    import atexit
    @atexit.register
    def cleanup():
        global cap
        if cap and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
    app.run(debug=False, use_reloader=False, threaded=True)