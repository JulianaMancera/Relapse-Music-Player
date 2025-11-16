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

app = Flask(__name__)

# Set up directories
script_dir = os.path.dirname(os.path.abspath(__file__))
music_dir = os.path.join(script_dir, "music")
lyrics_dir = os.path.join(script_dir, "lyrics")
static_dir = os.path.join(script_dir, "static")
playlist = [file for file in os.listdir(music_dir) if file.endswith(('.mp3', '.wav'))]

# Initialize Pygame mixer with error handling
try:
    pygame.mixer.init()
    pygame.mixer.music.set_volume(0.5)  # Set default volume to 50%
except pygame.error as e:
    logging.error(f"{datetime.now()}: Failed to initialize Pygame mixer - {e}")
    raise

current_song = None
current_index = 0
current_position = 0  # Track playback position in milliseconds
is_playing = False
is_camera_active = False  # Initialize camera as closed
last_gesture_time = 0  # For gesture debouncing
gesture_cooldown = 0.7  # Adjusted cooldown for stable gesture detection
current_gesture = None  # To persist gesture display
current_volume = 0.5  # Track current volume

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Function to find available cameras
def find_usb_camera():
    index = 0
    max_attempts = 10  # Limit the number of camera indices to check
    while index < max_attempts:
        temp_cap = cv2.VideoCapture(index)
        if temp_cap.isOpened():
            logging.info(f"{datetime.now()}: Found camera at index {index}")
            temp_cap.release()
            return index
        temp_cap.release()
        index += 1
    logging.error(f"{datetime.now()}: No USB camera found after checking {max_attempts} indices")
    return None

# Initialize camera (only when needed)
camera_index = find_usb_camera()
cap = None

# Setup logging
logging.basicConfig(filename='gesture_log.txt', level=logging.INFO)

# Gesture recognition logic
def recognize_gesture(landmarks):
    global last_gesture_time, current_gesture
    current_time = time.time()
    
    if current_time - last_gesture_time < gesture_cooldown:
        return None
    
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    # Play: All fingers up
    if (index_tip.y < landmarks[6].y - 0.03 and 
        middle_tip.y < landmarks[10].y - 0.03 and
        ring_tip.y < landmarks[14].y - 0.03 and 
        pinky_tip.y < landmarks[18].y - 0.03):
        last_gesture_time = current_time
        current_gesture = "play"
        return "play"
    
    # Pause: All fingers down
    if (index_tip.y > landmarks[6].y + 0.03 and 
        middle_tip.y > landmarks[10].y + 0.03 and
        ring_tip.y > landmarks[14].y + 0.03 and 
        pinky_tip.y > landmarks[18].y + 0.03):
        last_gesture_time = current_time
        current_gesture = "pause"
        return "pause"
    
    # Previous: Two fingers (index and middle) swiping left
    if (index_tip.x < wrist.x - 0.08 and middle_tip.x < wrist.x - 0.08 and
        abs(index_tip.y - middle_tip.y) < 0.03 and
        ring_tip.y > landmarks[14].y and pinky_tip.y > landmarks[18].y):
        last_gesture_time = current_time
        current_gesture = "previous"
        return "previous"
    
    # Next: Two fingers (index and middle) swiping right
    if (index_tip.x > wrist.x + 0.08 and middle_tip.x > wrist.x + 0.08 and
        abs(index_tip.y - middle_tip.y) < 0.03 and
        ring_tip.y > landmarks[14].y and pinky_tip.y > landmarks[18].y):
        last_gesture_time = current_time
        current_gesture = "next"
        return "next"
    
    # Volume Up: Thumb up, other fingers down
    if (thumb_tip.y < wrist.y - 0.06 and thumb_tip.x > wrist.x - 0.05 and thumb_tip.x < wrist.x + 0.05 and
        index_tip.y > landmarks[6].y + 0.04 and
        middle_tip.y > landmarks[10].y + 0.04 and
        ring_tip.y > landmarks[14].y + 0.04 and
        pinky_tip.y > landmarks[18].y + 0.04):
        last_gesture_time = current_time
        current_gesture = "volume_up"
        return "volume_up"
    
    # Volume Down: Thumb down, other fingers down
    if (thumb_tip.y > wrist.y + 0.06 and thumb_tip.x > wrist.x - 0.05 and thumb_tip.x < wrist.x + 0.05 and
        index_tip.y > landmarks[6].y + 0.04 and
        middle_tip.y > landmarks[10].y + 0.04 and
        ring_tip.y > landmarks[14].y + 0.04 and
        pinky_tip.y > landmarks[18].y + 0.04):
        last_gesture_time = current_time
        current_gesture = "volume_down"
        return "volume_down"
    
    current_gesture = None
    return None

# Video feed generator
def generate_video_feed():
    global is_playing, is_camera_active, cap, current_gesture
    while is_camera_active:
        if not cap or not cap.isOpened():
            logging.error(f"{datetime.now()}: Webcam not available")
            break
        success, frame = cap.read()
        if not success:
            logging.error(f"{datetime.now()}: Failed to read frame from webcam")
            break
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(frame_rgb)
        gesture = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = recognize_gesture(hand_landmarks.landmark)
                if gesture:
                    logging.info(f"{datetime.now()}: Detected gesture - {gesture} (thumb_tip: {hand_landmarks.landmark[4].x:.2f}, {hand_landmarks.landmark[4].y:.2f})")
                    handle_gesture(gesture)
        else:
            logging.info(f"{datetime.now()}: No hand landmarks detected")
        
        if current_gesture:
            cv2.putText(frame, f"Gesture: {current_gesture}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    blank_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', blank_frame)
    frame = buffer.tobytes()
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Handle gesture actions
def handle_gesture(gesture):
    global current_song, current_index, current_position, is_playing, current_volume
    if not playlist:
        logging.warning(f"{datetime.now()}: Playlist is empty")
        return
    if gesture == "play" and not pygame.mixer.music.get_busy():
        play_song()
        is_playing = True
    elif gesture == "pause" and pygame.mixer.music.get_busy():
        current_position = pygame.mixer.music.get_pos()
        pygame.mixer.music.pause()
        is_playing = False
    elif gesture == "next":
        next_song()
        is_playing = True
    elif gesture == "previous":
        previous_song()
        is_playing = True
    elif gesture == "volume_up":
        current_volume = min(1.0, current_volume + 0.02)  
        try:
            pygame.mixer.music.set_volume(current_volume)
            logging.info(f"{datetime.now()}: Volume increased to {current_volume:.2f}")
        except pygame.error as e:
            logging.error(f"{datetime.now()}: Failed to set volume - {e}")
    elif gesture == "volume_down":
        current_volume = max(0.0, current_volume - 0.00) 
        try:
            pygame.mixer.music.set_volume(current_volume)
            logging.info(f"{datetime.now()}: Volume decreased to {current_volume:.2f}")
        except pygame.error as e:
            logging.error(f"{datetime.now()}: Failed to set volume - {e}")

# Music control functions
def play_song():
    global current_song, current_index, current_position, is_playing, current_volume
    if not playlist:
        logging.warning(f"{datetime.now()}: No songs in playlist")
        return
    if 0 <= current_index < len(playlist):
        current_song = os.path.join(music_dir, playlist[current_index])
        try:
            pygame.mixer.music.load(current_song)
            pygame.mixer.music.set_volume(current_volume)  
            if current_position > 0:
                pygame.mixer.music.play(start=current_position / 1000.0)
                current_position = 0
            else:
                pygame.mixer.music.play()
            is_playing = True
            logging.info(f"{datetime.now()}: Playing {current_song} at {current_position}ms with volume {current_volume:.2f}")
        except pygame.error as e:
            logging.error(f"{datetime.now()}: Playback error - {e}")

def next_song():
    global current_index, current_position
    if not playlist:
        return
    current_position = 0
    current_index = (current_index + 1) % len(playlist)
    play_song()

def previous_song():
    global current_index, current_position
    if not playlist:
        return
    current_position = 0
    current_index = (current_index - 1) % len(playlist)
    play_song()

# Flask routes
@app.route('/')
def landing():
    return render_template('index.html')

@app.route('/player')
def player():
    return render_template('music_player_ui.html')

@app.route('/lyrics')
def lyrics():
    return render_template('lyrics.html')

@app.route('/music/<filename>')
def serve_music(filename):
    return send_from_directory(music_dir, filename)

@app.route('/lyrics/<filename>')
def serve_lyrics(filename):
    return send_from_directory(lyrics_dir, filename)

@app.route('/api/playlist')
def get_playlist():
    return jsonify(playlist)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control/<action>', methods=['POST'])
def control(action):
    global is_playing, is_camera_active, cap, camera_index
    if not playlist:
        return jsonify({'status': 'error', 'message': 'Playlist is empty'})
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
        if not is_camera_active:
            if cap and cap.isOpened():
                cap.release()
                cap = None
                logging.info(f"{datetime.now()}: Camera closed")
        else:
            if not cap:
                cap = cv2.VideoCapture(camera_index)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                    logging.info(f"{datetime.now()}: Camera reopened at index {camera_index}")
                else:
                    is_camera_active = False
                    logging.error(f"{datetime.now()}: Failed to reopen camera at index {camera_index}")
                    return jsonify({'status': 'error', 'message': 'Failed to reopen camera'})
        return jsonify({'status': 'success', 'is_camera_active': is_camera_active})
    return jsonify({'status': 'success'})

@app.route('/api/state')
def get_state():
    return jsonify({
        'current_index': current_index,
        'is_playing': is_playing,
        'is_camera_active': is_camera_active,
        'volume': round(current_volume, 2)  
    })

@app.route('/control/volume/<float:volume>', methods=['POST'])
def set_volume(volume):
    global current_volume
    if 0.0 <= volume <= 1.0:
        try:
            current_volume = volume
            pygame.mixer.music.set_volume(current_volume)
            logging.info(f"{datetime.now()}: Volume set to {current_volume:.2f}")
            return jsonify({'status': 'success', 'volume': round(current_volume, 2)})
        except pygame.error as e:
            logging.error(f"{datetime.now()}: Failed to set volume - {e}")
            return jsonify({'status': 'error', 'message': 'Failed to set volume'})
    else:
        logging.warning(f"{datetime.now()}: Invalid volume value {volume}")
        return jsonify({'status': 'error', 'message': 'Volume must be between 0.0 and 1.0'})

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
        logging.info(f"{datetime.now()}: Application cleanup completed")
    app.run(debug=False, use_reloader=False)