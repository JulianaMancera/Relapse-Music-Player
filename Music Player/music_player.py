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

app = Flask(__name__)

# DIRECTORIES
script_dir = os.path.dirname(os.path.abspath(__file__))
music_dir = os.path.join(script_dir, "music")
lyrics_dir = os.path.join(script_dir, "lyrics")
song_durations = {}
rickroll_triggered = False

# PLAYLIST
def load_playlist():
    if not os.path.exists(music_dir):
        return []
    songs = [f for f in os.listdir(music_dir) if f.lower().endswith(('.mp3', '.wav', '.ogg', '.flac', '.m4a'))]
    songs.sort(key=str.lower)
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

# SEARCH
search_buffer = ""
last_search_time = 0
SEARCH_TIMEOUT = 4.0

# MEDIAPIPE 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
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

# ASL LETTER RECOGNITION
def recognize_asl_letter(landmarks):
    tip = lambda i: landmarks.landmark[i]
    pip = lambda i: landmarks.landmark[i - 2 if i > 4 else i]  # Approximate for thumb

    def is_extended(finger_tip_id):
        return tip(finger_tip_id).y < tip(finger_tip_id - 2).y - 0.04  # Tip significantly above PIP

    def is_curled(finger_tip_id):
        return tip(finger_tip_id).y > tip(finger_tip_id - 2).y + 0.04

    def dist(a_id, b_id):
        a = tip(a_id)
        b = tip(b_id)
        return ((a.x - b.x)**2 + (a.y - b.y)**2)**0.5

    # Thumb tip (4), index tip (8), middle (12), ring (16), pinky (20)

    # A: Closed fist, thumb on side (all fingers curled, thumb not inside)
    if all(is_curled(fid) for fid in [8,12,16,20]) and tip(4).x > tip(5).x:  # Thumb outside index MCP
        return 'A'

    # B: Open palm, four fingers extended together, thumb folded in
    if (all(is_extended(fid) for fid in [8,12,16,20]) and
        not is_extended(4) and  # Thumb curled
        abs(tip(8).x - tip(20).x) < 0.15):  # Fingers close together
        return 'B'

    # C: Curved open hand forming C shape
    if (dist(4, 8) > 0.15 and  # Thumb and index far apart
        all(tip(fid).y > tip(0).y for fid in [8,12,16,20]) and  # Fingers below wrist (curved down)
        is_extended(4) and all(is_extended(fid) for fid in [8,12,16,20])):
        return 'C'

    # D: Index extended, others curled, thumb touching middle finger side
    if (is_extended(8) and all(is_curled(fid) for fid in [12,16,20]) and
        dist(4, 12) < 0.1):  # Thumb close to middle tip/PIP
        return 'D'

    # E: All fingers curled over thumb, flat-ish
    if (all(is_curled(fid) for fid in [8,12,16,20]) and
        tip(4).y > tip(8).y and  # Thumb tucked under fingers
        max(tip(fid).y for fid in [8,12,16,20]) - min(tip(fid).y for fid in [8,12,16,20]) < 0.08):  # Fingers at similar height
        return 'E'

    return None

# GESTURE RECOGNITION
def recognize_gesture(landmarks):
    global last_gesture_time, current_gesture, rickroll_triggered
    now = time.time()
    if now - last_gesture_time < gesture_cooldown:
        return None

    tip = lambda i: landmarks[i]

    # Finger states
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

def generate_video_feed():
    global is_camera_active, cap, current_gesture, rickroll_triggered

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
        
        if results.multi_hand_landmarks:
            hand = max(results.multi_hand_landmarks, key=lambda h: h.landmark[0].z)  # Pick the closest hand
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
                                      mp_drawing.DrawingSpec(color=(255,255,255), thickness=2))

            asl_letter = recognize_asl_letter(hand.landmark)
            gesture = recognize_gesture(hand.landmark)

            if asl_letter:
                cv2.putText(frame, f"ASL: {asl_letter}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
                log_gesture(f"ASL LETTER {asl_letter}")
                # Optional: print to console too
                print(f"Detected ASL letter: {asl_letter}")

            if gesture:
                handle_gesture(gesture)

        if current_gesture:
            txt = current_gesture.replace("_", " ").upper()
            cv2.putText(frame, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0), 4)
            cv2.putText(frame, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,100), 3)

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
    search_buffer = ""

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

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control/<action>', methods=['POST'])
def control(action):
    global is_camera_active
    if action == 'toggle_camera':
        is_camera_active = not is_camera_active
        if is_camera_active:
            open_camera() 
        return jsonify({'status': 'success', 'is_camera_active': is_camera_active})
    return jsonify({'status': 'success'})

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