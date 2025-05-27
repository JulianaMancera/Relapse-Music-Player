import pygame
import os
from pathlib import Path

pygame.init()
pygame.mixer.init()

# Set up the music directory dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
music_dir = os.path.join(script_dir, "music")
playlist = [file for file in os.listdir(music_dir) if file.endswith(('.mp3', '.wav'))]
current_track_index = 0
paused = False

# Function to load and play a track
def play_track(index):
    global current_track_index
    if 0 <= index < len(playlist):
        current_track_index = index
        track_path = os.path.join(music_dir, playlist[index])
        pygame.mixer.music.load(track_path)
        pygame.mixer.music.play()
        print(f"Now playing: {playlist[index]}")
    else:
        print("No more tracks in the playlist.")

# Function to pause/resume
def toggle_pause():
    global paused
    if paused:
        pygame.mixer.music.unpause()
        print("Resumed")
        paused = False
    else:
        pygame.mixer.music.pause()
        print("Paused")
        paused = True

# Function to stop playback
def stop_track():
    pygame.mixer.music.stop()
    print("Stopped")

# Function to play next track
def next_track():
    play_track(current_track_index + 1)

# Function to play previous track
def previous_track():
    play_track(current_track_index - 1)

# Main loop for user interaction
def main():
    global paused
    print("Simple Music Player")
    print("Commands: p (play), s (stop), n (next), b (previous), t (pause/resume), q (quit)")
    
    # Play the first track if available
    if playlist:
        play_track(current_track_index)
    else:
        print("No music files found in the music directory.")
        return

    while True:
        command = input("Enter command: ").lower()
        if command == 'p':
            play_track(current_track_index)
        elif command == 's':
            stop_track()
        elif command == 'n':
            next_track()
        elif command == 'b':
            previous_track()
        elif command == 't':
            toggle_pause()
        elif command == 'q':
            stop_track()
            pygame.mixer.quit()
            pygame.quit()
            print("Exiting music player.")
            break
        else:
            print("Invalid command. Use: p, s, n, b, t, q")

if __name__ == "__main__":
    main()