import os
from flask import Flask, render_template, jsonify, send_from_directory

app = Flask(__name__)

# Set up the music directory dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
music_dir = os.path.join(script_dir, "music")
playlist = [file for file in os.listdir(music_dir) if file.endswith(('.mp3', '.wav'))]

# Route for the landing page
@app.route('/')
def landing():
    return render_template('index.html')

# Route for the player UI
@app.route('/player')
def player():
    return render_template('music_player_ui.html')

# Route to serve music files
@app.route('/music/<filename>')
def serve_music(filename):
    return send_from_directory(music_dir, filename)

# API endpoint to get the playlist
@app.route('/api/playlist')
def get_playlist():
    return jsonify(playlist)

if __name__ == "__main__":
    app.run(debug=True)