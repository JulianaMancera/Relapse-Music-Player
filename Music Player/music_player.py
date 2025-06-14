import os
from flask import Flask, render_template, jsonify, send_from_directory, Response
import cv2
import mediapipe as mp
import threading
import pygame
import logging
from datetime import datetime

app = Flask(__name__)

# Set up directories
script_dir = os.path.dirname(os.path.abspath(__file__))
music_dir = os.path.join(script_dir, "music")
lyrics_dir = os.path.join(script_dir, "lyrics")
static_dir = os.path.join(script_dir, "static")
playlist = [file for file in os.listdir(music_dir) if file.endswith(('.mp3', '.wav'))]

# Initialize Pygame mixer
pygame.mixer.init()
current_song = None
current_index = 0
current_position = 0  # Track playback position in milliseconds
is_playing = False

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.85)
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Setup logging
logging.basicConfig(filename='gesture_log.txt', level=logging.INFO)

# Gesture recognition logic
def recognize_gesture(landmarks):
    wrist = landmarks[0]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    if (index_tip.y < landmarks[6].y and middle_tip.y < landmarks[10].y and
        ring_tip.y < landmarks[14].y and pinky_tip.y < landmarks[18].y):
        return "play"
    if (index_tip.y > landmarks[6].y and middle_tip.y > landmarks[10].y and
        ring_tip.y > landmarks[14].y and pinky_tip.y > landmarks[18].y):
        return "pause"
    if index_tip.x < wrist.x - 0.1:  # Pointing left for previous
        return "previous"
    if index_tip.x > wrist.x + 0.1:  # Pointing right for next
        return "next"
    return None

# Video feed generator
def generate_video_feed():
    global is_playing
    while True:
        success, frame = cap.read()
        if not success:
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
                    cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    logging.info(f"{datetime.now()}: Detected gesture - {gesture}")
                    handle_gesture(gesture)
        else:
            logging.info(f"{datetime.now()}: No hand landmarks detected")
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Handle gesture actions
def handle_gesture(gesture):
    global current_song, current_index, current_position, is_playing
    if gesture == "play" and not pygame.mixer.music.get_busy():
        play_song()
        is_playing = True
    elif gesture == "pause" and pygame.mixer.music.get_busy():
        current_position = pygame.mixer.music.get_pos()  # Save position in ms
        pygame.mixer.music.pause()
        is_playing = False
    elif gesture == "next":
        next_song()
        is_playing = True
    elif gesture == "previous":
        previous_song()
        is_playing = True

# Music control functions
def play_song():
    global current_song, current_index, current_position, is_playing
    if playlist and current_index < len(playlist):
        current_song = os.path.join(music_dir, playlist[current_index])
        try:
            pygame.mixer.music.load(current_song)
            if current_position > 0:
                pygame.mixer.music.play(start=current_position / 1000.0)  # Convert ms to seconds
                current_position = 0  # Reset after resuming
            else:
                pygame.mixer.music.play()
            is_playing = True
            logging.info(f"{datetime.now()}: Playing {current_song} at {current_position}ms")
        except Exception as e:
            logging.error(f"{datetime.now()}: Playback error - {e}")

def next_song():
    global current_index, current_position
    current_position = 0  # Reset position when switching songs
    if playlist:
        current_index = (current_index + 1) % len(playlist)
        play_song()

def previous_song():
    global current_index, current_position
    current_position = 0  # Reset position when switching songs
    if playlist:
        current_index = (current_index - 1) % len(playlist)
        play_song()

# Flask routes
@app.route('/')
def landing():
    return render_template('index.html')

@app.route('/player')
def player():
    return render_template('music_player_ui.html')

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
    global is_playing
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
    return jsonify({'status': 'success'})

@app.route('/api/state')
def get_state():
    return jsonify({
        'current_index': current_index,
        'is_playing': is_playing
    })

if __name__ == "__main__":
    import atexit
    @atexit.register
    def cleanup():
        cap.release()
        cv2.destroyAllWindows()
    app.run(debug=False, use_reloader=False)