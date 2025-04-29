import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
from pynput import keyboard
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class AudioFile:
    data: np.ndarray
    samplerate: int
    current_frame: int = 0

class AudioPlayer:
    def __init__(self, files: List[tuple[str, str]]):
        self.files = []
        self.file_to_play_by_channel = [0, 0]
        max_channels = 0
        for filepath in files:
            data, samplerate = sf.read(filepath)
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

    def callback(self, outdata, frames, time_info, status):
        if status:
            print(status)

        # Get chunks from all files
        chunks = []
        for file_index, file in enumerate(self.files):
            # Calculate remaining frames in current file
            remaining_frames = len(file.data) - file.current_frame
            
            if remaining_frames < frames:
                # Get remaining frames from current position
                chunk = file.data[file.current_frame:]
                # Get remaining frames from start of file
                remaining = file.data[:frames - remaining_frames]
                # Combine them
                chunk = np.vstack((chunk, remaining))
                file.current_frame = frames - remaining_frames
            else:
                chunk = file.data[file.current_frame:file.current_frame + frames]
                file.current_frame += frames
            
            # Handle channel count mismatch
            if chunk.shape[1] > self.channels:
                # Downmix to available channels
                if self.channels == 2:
                    # Simple stereo downmix: average channels 1-2 and 3-4
                    left = np.mean(chunk[:, :2], axis=1, keepdims=True)
                    right = np.mean(chunk[:, 2:], axis=1, keepdims=True)
                    chunk = np.hstack((left, right))
                else:
                    # For other channel counts, just take first N channels
                    chunk = chunk[:, :self.channels]
            elif chunk.shape[1] < self.channels:
                # Pad with zeros if needed
                chunk = np.pad(chunk, ((0, 0), (0, self.channels - chunk.shape[1])))
            
            for channel in range(2):
                if self.file_to_play_by_channel[channel] != file_index:
                    chunk[:, channel] = 0.0
                if self.file_to_play_by_channel[channel] != file_index:
                    chunk[:, channel] = 0.0
            chunks.append(chunk)

        # Mix all chunks
        mixed = np.sum(chunks, axis=0)
        # Normalize to prevent clipping
        if np.max(np.abs(mixed)) > 1.0:
            mixed = mixed / np.max(np.abs(mixed))

        outdata[:] = mixed

    def is_playing(self) -> bool:
        return any(file.current_frame < len(file.data) for file in self.files)

def list_audio_devices():
    print("\nAvailable audio devices:")
    print(sd.query_devices())
    
def main():
    # List available devices first
    list_audio_devices()
    
    # Define files to play with their toggle keys
    files = [
        'all.1.wav',
        'all.2.wav'git
    ]
    
    player = AudioPlayer(files)
    
    def on_press(key):
        try:
            if not key.char.isdigit():
                return
            channel = int(key.char)
            if channel >= len(player.files):
                return
            player.file_to_play_by_channel[channel] = (player.file_to_play_by_channel[channel] + 1) % 2
            print(f"files_to_play_by_channel {player.file_to_play_by_channel}")
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