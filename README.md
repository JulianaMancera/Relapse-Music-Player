<h1 align="center">Relapse Music Player</h1>
<div align="center">
  <img src="https://github.com/user-attachments/assets/c1d75d69-1080-4414-a97c-fe07f028dc57" width="500">
</div>

<h3 align="center"> A vinyl-themed web music player you can control with your hands. </h3>
&emsp;&emsp;Relapse Music Player is a Flask web app that plays the MP3 and WAV files in your local `music` folder. It uses your webcam and MediaPipe hand tracking so you can play, pause, and skip tracks without touching your keyboard or mouse. Sign-language (ASL) search is in development.
 
If a song takes you back to a forgotten summer, this player is here to help you relapse into the memory.
## ✨ Features
 
- 🎵 **Local playback:** MP3 and WAV files from a folder on your machine
- 🌐 **Browser UI:** browse your library, import songs, click a track to play
- 🤚 **Hand gesture controls:** play, pause, and skip hands-free (see below)
- 🔌 **Playlist API:** a small endpoint that returns your playlist as JSON
- ⚡ **Lightweight:** runs locally with minimal setup
### 🤚 Gesture controls
 
| Gesture | Action |
| --- | --- |
| ✋ Open palm | Play / Resume |
| ✊ Fist | Pause |
| ✌️ Two fingers, swipe left | Previous track |
| ✌️ Two fingers, swipe right | Next track |
 
**Tips for reliable detection**
 
- Use good, even lighting, and avoid strong backlight from a window.
- Keep your hand fully in frame and about an arm's length from the camera.
- Make each gesture deliberately and hold it briefly before moving on.
- Swipe at a steady pace rather than flicking.
### 🤟 ASL search (in development)
 
The goal is to search your library by fingerspelling a song title or artist with ASL letters in front of the webcam. Model training is still in progress, so this feature is not usable yet. See the [Roadmap](#-roadmap).
 
---
 
## 🧰 Tech stack
 
- **Backend:** Python, Flask
- **Computer vision:** OpenCV, MediaPipe
- **Audio:** pygame
- **Frontend:** HTML, CSS, JavaScript
---
 
## ⚙️ Getting started
 
### Prerequisites
 
- Python 3.9 or newer
- A working webcam (for gesture controls)
- Some MP3 or WAV files
### 1. Clone the repository
 
```bash
git clone https://github.com/JulianaMancera/Relapse-Music-Player.git
cd Relapse-Music-Player
```
 
### 2. Create a virtual environment (recommended)
 
```bash
python -m venv venv
 
# Windows
venv\Scripts\activate
 
# macOS / Linux
source venv/bin/activate
```
 
### 3. Install dependencies
 
```bash
pip install flask opencv-python mediapipe pygame
```
 
<!-- TODO: replace with `pip install -r requirements.txt` once a requirements.txt is added -->
 
### 4. Add your music
 
Create a folder named `music` in the project and put your MP3 or WAV files inside:
 
```
Relapse-Music-Player/
├── Music Player/
│   └── music_player.py
└── music/
    ├── song-one.mp3
    └── song-two.wav
```
<!-- TODO: confirm the real folder layout and where the app expects the `music` folder -->
 
### 5. Run the app
 
```bash
cd "Music Player"
python music_player.py
```
<!-- TODO: confirm the correct path to music_player.py -->
 
Open **http://127.0.0.1:5000** in your browser. Allow camera access if you want gesture controls.
 
---
 
## 🎮 Usage
 
**Web interface**
 
1. Browse your music collection on the main screen.
2. Import songs if you want to add more.
3. Click a track to start playing, and use the on-screen playback controls.
**Gesture controls**
 
1. Make sure your webcam is connected and not in use by another app.
2. Position yourself so your hand is clearly visible.
3. Use the gestures from the table above.
---
 
## 🛠️ Troubleshooting
 
| Problem | Try this |
| --- | --- |
| No songs show up | Check that your files are MP3 or WAV and are inside the `music` folder |
| Camera not detected | Close other apps that use the webcam, then restart the app |
| Gestures are missed or misread | Improve the lighting, move your hand closer, and hold the gesture a moment longer |
| Port 5000 is already in use | Stop the other process or change the port in the Flask `app.run(...)` call |
 
---
 
## 🗺️ Roadmap
 
- [x] MP3 / WAV playback in a web UI
- [x] Hand gesture controls (play, pause, previous, next)
- [ ] Make gesture detection more stable (debounce, cooldown, smoother swipes)
- [ ] ASL fingerspelling recognition model
- [ ] ASL-powered song search
- [ ] On-screen gesture feedback (detected gesture, camera status)
- [ ] Volume control, progress bar, and shuffle / repeat
- [ ] `requirements.txt` and a one-command setup
---
 
## 🤝 Contributing
 
Issues and pull requests are welcome. For larger changes, please open an issue first to discuss what you'd like to change.
 
## 👩🏼‍💻 Programmer
 
© 2025 Juliana Mancera
