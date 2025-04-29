import pygame
import time

# Initialize pygame mixer
pygame.mixer.init()

# Load the audio files
vocals = pygame.mixer.Sound("clips/numbered_split/1.vocals.wav")
music = pygame.mixer.Sound("clips/numbered_split/1.music.wav")

# Play both tracks simultaneously
vocals.play()
music.play()

# Keep the program running while the audio plays
# Get the length of the longer track
duration = max(vocals.get_length(), music.get_length())
time.sleep(duration)
