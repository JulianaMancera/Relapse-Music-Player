from flask import Flask, render_template, request, send_from_directory, jsonify
import os
from pathlib import Path

app = Flask(__name__, static_url_path='/static', static_folder='static')

script_dir = os.path.dirname(os.path.abspath(__file__))
music_dir = os.path.join(script_dir, "music")
playlist = [file for file in os.listdir(music_dir) if file.endswith(('.mp3', '.wav'))]
current_track_index = 0

@app.route('/')
def landing():
    return render_template('landing_page.html')

@app.route('/player')
def player():
    return render_template('music_player_ui.html', playlist=playlist)

@app.route('/music/<filename>')
def serve_music(filename):
    return send_from_directory(music_dir, filename)

@app.route('/next', methods=['POST'])
def next_track():
    global current_track_index
    if current_track_index < len(playlist) - 1:
        current_track_index += 1
    return jsonify({'current_track': playlist[current_track_index]})

@app.route('/prev', methods=['POST'])
def prev_track():
    global current_track_index
    if current_track_index > 0:
        current_track_index -= 1
    return jsonify({'current_track': playlist[current_track_index]})

@app.route('/add', methods=['POST'])
def add_song():
    new_song = request.json.get('song')
    if new_song and new_song not in playlist:
        playlist.append(new_song)
    return jsonify({'playlist': playlist})

if __name__ == "__main__":
    app.run(debug=True)