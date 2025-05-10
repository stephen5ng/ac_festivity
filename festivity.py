#!/usr/bin/env python3

import sounddevice as sd
import soundfile as sf
import os
import numpy as np
from pynput import keyboard
from dataclasses import dataclass
from typing import List, Dict

MUSIC_DIR = 'music'
CHANNELS = 4
FILES = [
    '1.wav',
    '2.wav',
    '3.wav',
    '4.wav'
]

@dataclass
class AudioFile:
    data: np.ndarray
    samplerate: int
    current_frame: int = 0

# Generate random play orders for each channel
channel_play_orders = [
    np.random.permutation(len(FILES)+1).tolist() for _ in range(CHANNELS)
]
print(f"channel_play_orders {channel_play_orders}")
class AudioPlayer:
    def __init__(self, files: List[tuple[str, str]]):
        self.files = []
        # Initialize indices to play file 4 for each channel by finding its position in each sequence
        self.index_to_play_by_channel = [
            channel_play_orders[channel].index(4) if 4 in channel_play_orders[channel] else 0
            for channel in range(CHANNELS)
        ]
        max_channels = 0
        for filepath in files:
            data, samplerate = sf.read(os.path.join(MUSIC_DIR, filepath))
            print(f"Loaded {filepath}: {data.shape[1]} channels at {samplerate} Hz")
            self.files.append(AudioFile(data=data, samplerate=samplerate))
            max_channels = max(max_channels, data.shape[1])
        
        self.samplerate = self.files[0].samplerate
        
        # Get default output device info
        device_info = sd.query_devices(kind='output')
        self.max_output_channels = device_info['max_output_channels']
        print(f"\nDefault output device supports {self.max_output_channels} channels")
        
        # Use minimum of input channels and device capabilities
        self.channels = min(max_channels, self.max_output_channels)
        print(f"Using {self.channels} output channels")

    def _handle_channel_mismatch(self, chunk: np.ndarray) -> np.ndarray:
        if chunk.shape[1] > self.channels:
            # print(f"Chunk has {chunk.shape[1]} channels, downmixing to {self.channels}")
            left = np.mean(chunk[:, :2], axis=1, keepdims=True)
            right = np.mean(chunk[:, 2:], axis=1, keepdims=True)
            chunk = np.hstack((left, right))
        elif chunk.shape[1] < self.channels:
            chunk = np.pad(chunk, ((0, 0), (0, self.channels - chunk.shape[1])))
        return chunk

    def _select_channels(self, chunk: np.ndarray, file_index: int) -> np.ndarray:
        # Make a copy of the chunk to avoid mutating the original data
        selected = np.zeros_like(chunk)
        for channel in range(CHANNELS):
            # print(f"channel {channel} index_to_play_by_channel {self.index_to_play_by_channel[channel]}")
            if channel_play_orders[channel][self.index_to_play_by_channel[channel]] == file_index:
                selected[:, channel] = chunk[:, channel]
        return selected

    def callback(self, outdata, frames, time_info, status):
        chunks = []
        for file_index, file in enumerate(self.files):
            remaining_frames = len(file.data) - file.current_frame
            
            if remaining_frames < frames:
                file.current_frame = 0

            chunk = file.data[file.current_frame:file.current_frame + frames]
            file.current_frame += frames
            
            chunk = self._select_channels(chunk, file_index)
            chunks.append(chunk)

        # Mix all chunks to get 4-channel sound
        mixed = np.sum(chunks, axis=0)
        
        # Then downmix to stereo if needed
        if outdata.shape[1] == 2:
            left = np.mean(mixed[:, :2], axis=1, keepdims=True)
            right = np.mean(mixed[:, 2:], axis=1, keepdims=True)
            mixed = np.hstack((left, right))
        
        if np.max(np.abs(mixed)) > 1.0:
            mixed = mixed / np.max(np.abs(mixed))

        outdata[:] = mixed

    def is_playing(self) -> bool:
        return any(file.current_frame < len(file.data) for file in self.files)

def list_audio_devices():
    print("\nAvailable audio devices:")
    print(sd.query_devices())
    
def main():
    list_audio_devices()
    
    player = AudioPlayer(FILES)
    def on_press(key):
        try:
            if not key.char.isdigit():
                return
            channel = int(key.char)
            if channel == 0 or channel >= CHANNELS+1:
                return
            
            # 0-index
            channel -= 1

            # Add a dummy extra file for silence.
            player.index_to_play_by_channel[channel] = ((player.index_to_play_by_channel[channel] + 1) 
                                                       % (1 + len(player.files)))
            print(f"index_to_play_by_channel {player.index_to_play_by_channel}")
            print(f"file_to_play_by_channel {[channel_play_orders[c][player.index_to_play_by_channel[c]] for c in range(4)]}")
        except AttributeError:
            return

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Start audio playback
    with sd.OutputStream(
        samplerate=player.samplerate,
        channels=player.channels,
        dtype='float32',
        callback=player.callback
    ):
        print("\nPlaying audio...")
        while player.is_playing():
            sd.sleep(100)

    print("Done!")
    listener.stop()

if __name__ == "__main__":
    main() 