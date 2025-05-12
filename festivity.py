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
    '0.wav',
    '1.wav',
    '2.wav',
    '3.wav'
]
CHANNEL_VOLUME = [1, 1, 0.2, 0.1]
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
        print(f"index_to_play_by_channel {self.index_to_play_by_channel}")
        self.control_channel = 0
        for filepath in files:
            data, samplerate = sf.read(os.path.join(MUSIC_DIR, filepath))
            if data.shape[1] != CHANNELS:
                raise ValueError(f"File {filepath} has {data.shape[1]} channels, expected {CHANNELS}")
            print(f"Loaded {filepath}: {data.shape[1]} channels at {samplerate} Hz")
            self.files.append(AudioFile(data=data, samplerate=samplerate))
        
        self.samplerate = self.files[0].samplerate
        self.matched_files = []
        
        # Get default output device info
        device_info = sd.query_devices(kind='output')
        self.max_output_channels = device_info['max_output_channels']
        print(f"\nDefault output device supports {self.max_output_channels} channels")
        
        # Use all available output channels
        self.channels = self.max_output_channels
        print(f"Using {self.channels} output channels")

    def _handle_channel_mismatch(self, chunk: np.ndarray) -> np.ndarray:
        # Create a new array with all output channels, initialized to silence
        output = np.zeros((chunk.shape[0], self.channels))
        # Copy the available channels from the input
        channels_to_copy = min(chunk.shape[1], self.channels)
        output[:, :channels_to_copy] = chunk[:, :channels_to_copy]
        return output

    def _select_channels(self, chunk: np.ndarray, file_index: int) -> np.ndarray:
        # Make a copy of the chunk to avoid mutating the original data
        selected = np.zeros_like(chunk)
        for channel in range(CHANNELS):
            # print(f"channel {channel} index_to_play_by_channel {self.index_to_play_by_channel[channel]}")
            if channel_play_orders[channel][self.index_to_play_by_channel[channel]] == file_index:
                selected[:, channel] = chunk[:, channel]
        return selected

    def _adjust_channel_volumes(self, mixed: np.ndarray) -> np.ndarray:
        """Adjust the volume of each channel according to CHANNEL_VOLUME.
        Uses logarithmic scaling to account for human perception of sound."""
        for channel in range(CHANNELS):
            # Convert percentage to 0-1 range and apply logarithmic scaling
            volume_db = 20 * np.log10(CHANNEL_VOLUME[channel])
            # Convert back to linear scale for actual multiplication
            volume_linear = 10 ** (volume_db / 20)
            mixed[:, channel] *= volume_linear
        return mixed

    def _mix_if_same_file(self, mixed: np.ndarray) -> tuple[np.ndarray, bool]:
        """If all channels are playing the same file, mix them together and copy to all output channels.
        Returns a tuple of (mixed audio array, whether files were mixed)."""
        current_files = [channel_play_orders[channel][self.index_to_play_by_channel[channel]] 
                        for channel in range(CHANNELS)]
        if len(set(current_files)) == 1 and current_files[0] != 4:  # Don't mix if all channels are silent (file 4)
            # Mix all channels together
            channel_mix = np.mean(mixed[:, :CHANNELS], axis=1, keepdims=True)
            # Apply the mix to all available output channels
            mixed = np.tile(channel_mix, (1, mixed.shape[1]))
            return mixed, True
        return mixed, False

    def _downmix_to_stereo_if_needed(self, mixed: np.ndarray, output_channels: int) -> np.ndarray:
        """Downmix to stereo if the output device only supports 2 channels.
        Returns the downmixed audio array."""
        if output_channels == 2:
            left = np.mean(mixed[:, :2], axis=1, keepdims=True)
            right = np.mean(mixed[:, 2:], axis=1, keepdims=True)
            mixed = np.hstack((left, right))
        return mixed

    def _normalize_volume(self, mixed: np.ndarray) -> np.ndarray:
        """Normalize the volume if it exceeds 1.0 to prevent clipping.
        Returns the normalized audio array."""
        if np.max(np.abs(mixed)) > 1.0:
            mixed = mixed / np.max(np.abs(mixed))
        return mixed

    def _process_chunks(self, frames: int) -> List[np.ndarray]:
        """Process audio chunks for all files, handling looping and channel selection.
        Returns a list of processed audio chunks."""
        chunks = []
        for file_index, file in enumerate(self.files):
            remaining_frames = len(file.data) - file.current_frame
            
            if remaining_frames < frames:
                file.current_frame = 0

            chunk = file.data[file.current_frame:file.current_frame + frames]
            file.current_frame += frames
            
            chunk = self._select_channels(chunk, file_index)
            chunks.append(chunk)
        return chunks

    def callback(self, outdata, frames, time_info, status):
        control_channel = int(time_info.outputBufferDacTime) % CHANNELS
        if control_channel != self.control_channel:
            self.control_channel = control_channel
            print(f"control_channel: {control_channel}")

        chunks = self._process_chunks(frames)

        # Mix all chunks to get 4-channel sound
        mixed = np.sum(chunks, axis=0)
        
        mixed, is_mixed = self._mix_if_same_file(mixed)
        
        # Only adjust volumes if we're not mixing (i.e., different files on different channels)
        if not is_mixed:
            mixed = self._adjust_channel_volumes(mixed)
            
        # Expand to all output channels if needed
        if mixed.shape[1] != outdata.shape[1]:
            if is_mixed:
                # If mixed, copy the first channel to all output channels
                mixed = np.tile(mixed[:, :1], (1, outdata.shape[1]))
            else:
                # If not mixed, keep first CHANNELS and silence the rest
                output = np.zeros((mixed.shape[0], outdata.shape[1]))
                output[:, :mixed.shape[1]] = mixed
                mixed = output

        mixed = self._normalize_volume(mixed)
        outdata[:] = mixed

def list_audio_devices():
    print("\nAvailable audio devices:")
    print(sd.query_devices())
    
def main():
    list_audio_devices()
    
    player = AudioPlayer(FILES)
    def on_press(key):
        try:
            channel = None
            if key == keyboard.Key.space:
                channel = player.control_channel
            elif hasattr(key, 'char'):
                if key.char == 's':
                    for file in player.files:
                        file.current_frame = 0
                    print("Restarted all files from beginning")
                    return
                elif key.char.isdigit():
                    channel = int(key.char) - 1
                    if channel < 0 or channel >= CHANNELS:
                        return
            
            if channel is not None:
                # Add a dummy extra file for silence.
                player.index_to_play_by_channel[channel] = ((player.index_to_play_by_channel[channel] + 1) 
                                                           % (1 + len(player.files)))
                files_to_play_by_channel = [channel_play_orders[c][player.index_to_play_by_channel[c]] for c in range(4)]
                if len(set(files_to_play_by_channel)) == 1:
                    print("winner: restarting")
                    for file in player.files:
                        file.current_frame = 0
                print(f"index_to_play_by_channel {player.index_to_play_by_channel}")
                print(f"files_to_play_by_channel {files_to_play_by_channel}")
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
        while True:
            sd.sleep(100)

    print("Done!")
    listener.stop()

if __name__ == "__main__":
    main() 