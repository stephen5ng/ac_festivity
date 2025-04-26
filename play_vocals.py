import pygame
import time

# Initialize pygame mixer
pygame.mixer.init()

# Load and play the audio file
pygame.mixer.music.load('clips/merged/vocals.wav')
pygame.mixer.music.play()

# Keep the script running while the music plays
while pygame.mixer.music.get_busy():
    time.sleep(1) 