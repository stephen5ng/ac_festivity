#!/usr/bin/env python3

"""A multi-channel audio player that allows synchronized playback of multiple audio files.

This program creates a 4-channel audio system where each channel can independently play
different audio files. Players can control their assigned channel using number keys (1-4)
or a GPIO button (which rotates through channels automatically). When all channels
synchronize on the same file, that file is considered a "winner" and is removed from
future play options after it finishes playing.

Usage:
    Run the script and use number keys 1-4 to control each channel, or GPIO button to
    control the currently active channel. Press 's' to restart all files from the beginning.
"""

from collections import Counter
import sounddevice as sd
import soundfile as sf
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum, auto
import sys
import threading
import queue
import time
import platform
import traceback
import select

SECONDS_TO_ANNOUNCE_CHANNEL = 2
# GPIO Configuration - only import and use on Raspberry Pi
IS_RASPBERRY_PI = platform.system() == 'Linux' and platform.machine().startswith('aarch')
if IS_RASPBERRY_PI:
    try:
        import RPi.GPIO as GPIO
        # Using BCM numbering (GPIO numbers)
        NEXT_FILE_BUTTON_PIN = 17
        WIN_BUTTON_PIN = 27
        NEXT_CHANNEL_BUTTON_PIN = 22
        DEBOUNCE_TIME = 0.2      # Button debounce time in seconds
    except ImportError:
        print("Warning: RPi.GPIO not available. Running without GPIO support.")
        IS_RASPBERRY_PI = False

def setup_gpio():
    """Initialize GPIO pins if on Raspberry Pi."""
    if not IS_RASPBERRY_PI:
        return False
        
    try:
        GPIO.cleanup()
        GPIO.setmode(GPIO.BCM)  # Use BCM numbering
        GPIO.setup(NEXT_FILE_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(WIN_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(NEXT_CHANNEL_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        print(f"GPIO setup successful on GPIO{NEXT_FILE_BUTTON_PIN}, GPIO{WIN_BUTTON_PIN}, and GPIO{NEXT_CHANNEL_BUTTON_PIN}")
        return True
    except Exception as e:
        print(f"Error setting up GPIO: {e}")
        return False

class PlayerState(Enum):
    IDLE = auto()
    CHANGE_CHANNEL = auto()
    PLAYING_CHANNEL_ANNOUNCEMENT = auto()
    PLAY_MATCH_ANNOUNCEMENT = auto()
    PLAYING_MATCH_ANNOUNCEMENT = auto()
    PLAY_VICTORY_ANNOUNCEMENT = auto()
    PLAYING_VICTORY_ANNOUNCEMENT = auto()
    PLAY_VICTORY_FILE = auto()
    PLAYING_VICTORY_FILE = auto()
    PLAYED_VICTORY_FILE = auto()
    PLAY_NEXT_FILE_ANNOUNCEMENT = auto()
    PLAYING_NEXT_FILE_ANNOUNCEMENT = auto()

MUSIC_DIR = 'music'
VOICE_DIR = 'voices'
CHANNELS = 4
SILENT_CHANNEL = 0
SONG_FILES = [
    '0.wav',
    '1.wav',
    '2.wav',
    '3.wav',
    '4.wav',
    '5.wav'
]
FULL_SONG_FILES = [
    '1.full.wav',
    '2.full.wav',
    '3.full.wav',
    '4.full.wav',
    '5.full.wav'
]

CHANNEL_ANNOUNCE_FILES = [
    'one.wav',
    'two.wav',
    'three.wav',
    'four.wav'
]
CHANNEL_MATCH_FILES = [
    '0_matches.wav',
    '1_matches.wav',
    '2_matches.wav',
    '3_matches.wav',
]
VICTORY_SOUND_FILE = 'win.wav'
DING_SOUND_FILE = 'ding.wav'
FILE_COUNT = len(SONG_FILES)
NEXT_FILE_FILE = 'next_song.wav'
CHANNEL_VOLUME = [1, 1, 1, 1]
@dataclass
class AudioFile:
    data: np.ndarray
    samplerate: int
    current_frame: int = 0

# Generate random play orders for each channel
channel_play_orders = [
    np.random.permutation(len(SONG_FILES)).tolist() for _ in range(CHANNELS)
]
print(f"channel_play_orders {channel_play_orders}")

class AudioPlayer:
    def __init__(self):
        self.files = []
        self.full_files = []  # Store full song files
        # Initialize indices to play silence for each channel by finding its position in each sequence
        self.index_to_play_by_channel = [
            channel_play_orders[channel].index(SILENT_CHANNEL)
            for channel in range(CHANNELS)
        ]
        print(f"index_to_play_by_channel {self.index_to_play_by_channel}")
        self.control_channel = 0
        self.duplicate_count = None
        self.winning_file = None  # Track which file has won but not finished playing
        self.done_playing_announcement = False  # Track if current announcement is done playing
        self.state = PlayerState.IDLE
        self.playing_victory_announcement = False
        self.announcement_start_time = 0  # Track when announcement started
        self.should_exit = False  # Flag to indicate when all songs are matched
        
        # Load game files
        self.files = self._load_audio_files(SONG_FILES)
        self.full_files = self._load_audio_files(FULL_SONG_FILES)
        
        # Load channel announcement files
        self.channel_announce_files = []
        self.channel_announce_files = self._load_voice_files(CHANNEL_ANNOUNCE_FILES)
        self.channel_announce_file = self.channel_announce_files[0]  # Start with "one.wav"

        # Load channel match files
        self.channel_match_files = self._load_voice_files(CHANNEL_MATCH_FILES)

        # Load victory sound
        victory_data, victory_samplerate = sf.read(os.path.join(VOICE_DIR, VICTORY_SOUND_FILE))
        if victory_samplerate != self.files[0].samplerate:
            raise ValueError(f"{VICTORY_SOUND_FILE} must have same sample rate as game files")
        self.victory_file = AudioFile(data=victory_data, samplerate=victory_samplerate)
        
        next_file_data, next_file_data_samplerate = sf.read(os.path.join(VOICE_DIR, NEXT_FILE_FILE))
        if next_file_data_samplerate != self.files[0].samplerate:
            raise ValueError(f"{NEXT_FILE_FILE} must have same sample rate as game files")
        self.next_file_file = AudioFile(data=next_file_data, samplerate=next_file_data_samplerate)

        self.samplerate = self.files[0].samplerate
        self.matched_files = []
        
        # Get default output device info with fallback for Raspberry Pi
        try:
            # List all devices
            devices = sd.query_devices()
            print("\nAvailable audio devices:")
            for i, dev in enumerate(devices):
                print(f"{i}: {dev['name']} (in={dev['max_input_channels']}, out={dev['max_output_channels']})")
            
            # Try to find USB audio device
            device_id = None
            for i, dev in enumerate(devices):
                if 'USB Audio' in dev['name'] or 'USB Sound' in dev['name']:
                    device_id = i
                    print(f"\nUsing USB Audio device {device_id}")
                    break
            
            if device_id is None:
                print("\nUSB Audio device not found, using default device")
            
            # Test the device with a simple tone
            print("\nTesting audio device...")
            duration = 0.5  # seconds
            t = np.linspace(0, duration, int(self.samplerate * duration), False)
            tone = 0.5 * np.sin(2 * np.pi * 440 * t)
            stereo = np.column_stack((tone, tone))
            sd.play(stereo, self.samplerate, device=device_id)
            sd.wait()
            print("Audio device test successful!")
            
            # Set as default device
            sd.default.device = device_id
            device_info = sd.query_devices(device_id if device_id is not None else sd.default.device[1])
            self.max_output_channels = device_info['max_output_channels']
            print(f"Device supports {self.max_output_channels} output channels")
            
        except Exception as e:
            print(f"\nError initializing audio: {str(e)}")
            print("\nPlease check your audio configuration:")
            print("1. Run 'aplay -l' to list audio devices")
            print("2. Make sure your audio output is enabled in raspi-config")
            print("3. Try setting the default device with:")
            print("   export AUDIODEV=hw:0,0  # USB Audio device")
            raise SystemExit(1)
        
        self.channels = min(6, self.max_output_channels)
        print(f"Using {self.channels} output channels")
        # Sleep to allow time to display output
        time.sleep(2)

    def _load_audio_files(self, file_list: List[str]) -> List[AudioFile]:
        """Load a list of audio files and return them as AudioFile objects.
        
        Args:
            file_list: List of filenames to load
            
        Returns:
            List of loaded AudioFile objects
            
        Raises:
            ValueError: If any file has incorrect number of channels
        """
        loaded_files = []
        for filepath in file_list:
            data, samplerate = sf.read(os.path.join(MUSIC_DIR, filepath))
            print(f"Loaded {filepath}: {data.shape} channels at {samplerate} Hz")
            if data.shape[1] != CHANNELS:
                raise ValueError(f"File {filepath} has {data.shape[1]} channels, expected {CHANNELS}")
            print(f"Loaded {filepath}: {data.shape[1]} channels at {samplerate} Hz")
            loaded_files.append(AudioFile(data=data, samplerate=samplerate))
        return loaded_files

    def _load_voice_files(self, file_list: List[str]) -> List[AudioFile]:
        """Load a list of voice files from VOICE_DIR and return them as AudioFile objects.
        
        Args:
            file_list: List of filenames to load from VOICE_DIR
            
        Returns:
            List of loaded AudioFile objects
            
        Raises:
            ValueError: If any file has wrong sample rate
        """
        loaded_files = []
        for filepath in file_list:
            data, samplerate = sf.read(os.path.join(VOICE_DIR, filepath))
            if samplerate != self.files[0].samplerate:
                raise ValueError(f"{filepath} must have same sample rate as game files")
            print(f"Loaded voice {filepath}: {data.shape} channels at {samplerate} Hz")
            loaded_files.append(AudioFile(data=data, samplerate=samplerate))
        return loaded_files

    @property
    def is_victory_state(self) -> bool:
        """Whether the player is in a victory file state."""
        return self.state in [PlayerState.PLAY_VICTORY_FILE, PlayerState.PLAYING_VICTORY_FILE]

    def _select_channels(self, chunk: np.ndarray, file_index: int) -> np.ndarray:
        # Make a copy of the chunk to avoid mutating the original data
        selected = np.zeros_like(chunk)
        for channel in range(CHANNELS):
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

    def _mix_to_all_channels(self, mixed: np.ndarray) -> np.ndarray:
        """Mix all channels together and copy to all output channels.
        Returns the mixed audio array."""
        # Mix all channels together
        channel_mix = np.mean(mixed[:, :CHANNELS], axis=1, keepdims=True)
        # Apply the mix to all available output channels
        return np.tile(channel_mix, (1, mixed.shape[1]))

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

    def _process_song_chunks(self, frames: int) -> List[np.ndarray]:
        """Process audio chunks for all files, handling looping and channel selection.
        
        During normal play, uses fragment files. During victory, uses full song for winning file.
        
        Args:
            frames: Number of frames to process
            
        Returns:
            List of processed audio chunks, one per file
        """
        chunks = []
        
        # Get the appropriate file list for each index
        files_to_use = self.full_files if self.is_victory_state else self.files
        
        for file_index in range(len(self.files)):
            # Skip non-winning files during victory
            if self.is_victory_state and file_index != self.winning_file:
                continue
                
            # Get the appropriate file (full or fragment)
            file = files_to_use[file_index]
            
            # Handle file end
            remaining_frames = len(file.data) - file.current_frame
            if remaining_frames < frames:
                if not self.is_victory_state:
                    file.current_frame = 0
                else:
                    continue

            # Get chunk and update position
            chunk = file.data[file.current_frame:file.current_frame + frames]
            file.current_frame += frames
            
            # Select channels for this file
            chunk = self._select_channels(chunk, file_index)
            chunks.append(chunk)
            
        return chunks

    def _next_announcement_state(self, state: PlayerState):
        if state == PlayerState.PLAYING_CHANNEL_ANNOUNCEMENT:
            if time.time() < self.announcement_start_time + SECONDS_TO_ANNOUNCE_CHANNEL:
                return PlayerState.PLAYING_CHANNEL_ANNOUNCEMENT
            return PlayerState.IDLE
        elif state == PlayerState.PLAYING_MATCH_ANNOUNCEMENT:
            return PlayerState.IDLE
        elif state == PlayerState.PLAYING_VICTORY_ANNOUNCEMENT:
            return PlayerState.PLAY_VICTORY_FILE
        elif state == PlayerState.PLAYING_VICTORY_FILE:
            return PlayerState.PLAYED_VICTORY_FILE
        elif state == PlayerState.PLAYED_VICTORY_FILE:
            return PlayerState.IDLE
        elif state == PlayerState.PLAYING_NEXT_FILE_ANNOUNCEMENT:
            return PlayerState.IDLE
        return PlayerState.IDLE

    def callback(self, outdata, frames, time_info, status):
        if self.state == PlayerState.IDLE:
            # Do nothing in IDLE state - wait for explicit channel change command
            pass
        elif self.state == PlayerState.CHANGE_CHANNEL:
            self.control_channel += 1
            self.control_channel %= CHANNELS
            print(f"control_channel: {self.control_channel+1}")
            self.channel_announce_file = self.channel_announce_files[self.control_channel]
            self.channel_announce_file.current_frame = 0
            self.state = PlayerState.PLAYING_CHANNEL_ANNOUNCEMENT
            self.announcement_start_time = time.time()
        elif self.state == PlayerState.PLAY_NEXT_FILE_ANNOUNCEMENT:
            self.channel_announce_file = self.next_file_file
            self.channel_announce_file.current_frame = 0
            self.state = PlayerState.PLAYING_NEXT_FILE_ANNOUNCEMENT
        elif self.state == PlayerState.PLAY_MATCH_ANNOUNCEMENT:
            self.channel_announce_file = self.channel_match_files[self.duplicate_count]
            self.state = PlayerState.PLAYING_MATCH_ANNOUNCEMENT
            self.channel_announce_file.current_frame = 0
        elif self.state == PlayerState.PLAY_VICTORY_ANNOUNCEMENT:
            self.channel_announce_file = self.victory_file
            self.channel_announce_file.current_frame = 0
            self.state = PlayerState.PLAYING_VICTORY_ANNOUNCEMENT
        elif self.state == PlayerState.PLAY_VICTORY_FILE:
            self.state = PlayerState.PLAYING_VICTORY_FILE
            self.full_files[self.winning_file].current_frame = 0
        elif self.state == PlayerState.PLAYED_VICTORY_FILE:
            self._handle_win(self.winning_file)
            self.state = PlayerState.IDLE

        chunks = self._process_song_chunks(frames)
        if not chunks:
            outdata.fill(0)
            return

        # Mix all chunks to get 4-channel sound
        mixed = np.sum(chunks, axis=0)
        
        # Pad to match output device channels
        padded = np.zeros((mixed.shape[0], outdata.shape[1]))
        padded[:, :mixed.shape[1]] = mixed
        mixed = padded

        # Get announcement chunk
        if self.channel_announce_file.current_frame >= len(self.channel_announce_file.data):
            channel_announce_chunk = np.zeros((frames,))
        else:
            channel_announce_chunk = self.channel_announce_file.data[self.channel_announce_file.current_frame:
                self.channel_announce_file.current_frame + frames]
            if len(channel_announce_chunk) < frames:
                channel_announce_chunk = np.pad(channel_announce_chunk, (0, frames - len(channel_announce_chunk)))

        self.channel_announce_file.current_frame += frames

        # Check if victory file has finished playing
        if self.is_victory_state:
            remaining_frames = len(self.full_files[self.winning_file].data) - self.full_files[self.winning_file].current_frame
            
            if remaining_frames <= frames:
                self.state = self._next_announcement_state(self.state)
        elif self.channel_announce_file.current_frame >= len(self.channel_announce_file.data):
            self.state = self._next_announcement_state(self.state)

        # Handle all mixing in one place
        if self.is_victory_state:
            mixed = self._mix_to_all_channels(mixed)
        else:
            mixed = self._adjust_channel_volumes(mixed)
            # Play announcements on channel 5 (index 4)
            if mixed.shape[1] > 4:
                mixed[:, 4] += channel_announce_chunk * 0.5  # Channel 5
            # Fallback to channel 1 if 5 not available
            else:
                mixed[:, 0] += channel_announce_chunk * 0.5  # Channel 1

        # Copy announcement to first four channels during victory states
        if self.state in [PlayerState.PLAYING_VICTORY_ANNOUNCEMENT]:
            for i in range(min(4, mixed.shape[1])):  # copy to first 4 channels
                mixed[:, i] = channel_announce_chunk

        mixed = self._normalize_volume(mixed)
        outdata[:] = mixed

    def _handle_win(self, winning_file: int) -> None:
        """Remove the winning file from all play orders and reset channels to silent track."""
        # Remove winning file from all play orders
        for i in range(len(channel_play_orders)):
            channel_play_orders[i] = [x for x in channel_play_orders[i] if x != winning_file]
            # Reset to silent track
            self.index_to_play_by_channel = [
                channel_play_orders[channel].index(SILENT_CHANNEL) for channel in range(CHANNELS)
            ]
            
        # Check if there are any songs remaining (excluding the silent track)
        remaining_songs = [x for x in channel_play_orders[SILENT_CHANNEL] if x != 0]
        if not remaining_songs:
            print("\nAll songs have been matched! Game complete.")
            self.should_exit = True

def list_audio_devices():
    print("\nAvailable audio devices:")
    print(sd.query_devices())

def max_duplicate_count(nums):
    counts = Counter(nums)
    duplicate_counts = [count for count in counts.values() if count > 1]
    return max(duplicate_counts) if duplicate_counts else 0

def main():
    try:
        # Initialize GPIO first if on Raspberry Pi
        use_gpio = False
        if IS_RASPBERRY_PI:
            if setup_gpio():
                use_gpio = True
                print("GPIO setup successful")
            else:
                print("Failed to initialize GPIO. Running without button support.")
        else:
            print("Running without GPIO support (not on Raspberry Pi)")
            
        try:
            player = AudioPlayer()
        except Exception as e:
            print(f"Error initializing audio: {str(e)}")
            traceback.print_exc()
            raise SystemExit(1)
            
        input_queue = queue.Queue()
        last_channel_button_press = 0
        last_win_button_press = 0
        last_channel_state = GPIO.HIGH if IS_RASPBERRY_PI else None
        last_win_state = GPIO.HIGH if IS_RASPBERRY_PI else None
        
        def gpio_poll_thread():
            """Thread to poll GPIO buttons on Raspberry Pi."""
            if not IS_RASPBERRY_PI:
                return
                
            nonlocal last_channel_state, last_win_state, last_channel_button_press, last_win_button_press
            last_next_channel_state = GPIO.HIGH
            last_next_channel_press = 0
            
            while True:
                try:
                    # Read current states
                    next_file_state = GPIO.input(NEXT_FILE_BUTTON_PIN)
                    win_state = GPIO.input(WIN_BUTTON_PIN)
                    next_channel_state = GPIO.input(NEXT_CHANNEL_BUTTON_PIN)
                    current_time = time.time()
                    
                    # Check if all buttons are pressed
                    if next_file_state == GPIO.LOW and win_state == GPIO.LOW and next_channel_state == GPIO.LOW:
                        print("All buttons pressed - exiting game!")
                        input_queue.put('exit')
                        break
                    
                    # Check for next file button press (FALLING edge)
                    if next_file_state == GPIO.LOW and last_channel_state == GPIO.HIGH:
                        if current_time - last_channel_button_press > DEBOUNCE_TIME:
                            input_queue.put('f')
                            last_channel_button_press = current_time
                            print("Next file button pressed")
                    
                    # Check for next channel button press (FALLING edge)
                    if next_channel_state == GPIO.LOW and last_next_channel_state == GPIO.HIGH:
                        if current_time - last_next_channel_press > DEBOUNCE_TIME:
                            input_queue.put('c')
                            last_next_channel_press = current_time
                            print("Next channel button pressed")
                    
                    # Check for win button press (FALLING edge)
                    if win_state == GPIO.LOW and last_win_state == GPIO.HIGH:
                        if current_time - last_win_button_press > DEBOUNCE_TIME:
                            input_queue.put('w')
                            last_win_button_press = current_time
                            print("Win button pressed")
                    
                    # Update last states
                    last_channel_state = next_file_state
                    last_win_state = win_state
                    last_next_channel_state = next_channel_state
                    
                    # Small sleep to prevent CPU hogging
                    time.sleep(0.1)
                except Exception as e:
                    print(f"Error in GPIO poll thread: {e}")
                    break
        
        # Start GPIO poll thread if on Raspberry Pi and GPIO is available
        if use_gpio:
            gpio_thread = threading.Thread(target=gpio_poll_thread, daemon=True)
            gpio_thread.start()
            print("GPIO polling started")
        
        # Start audio playback
        with sd.OutputStream(
            samplerate=player.samplerate,
            channels=player.channels,
            dtype='float32',
            callback=player.callback
        ):
            print("\n=== Game Controls ===")
            print("f: Next file in current channel")
            print("c: Change to next channel")
            print("w: Check for win")
            print("s: Restart all files")
            print("1-4: Select channel")
            print("Ctrl+C: Exit")
            print("===================\n")
            
            while True:
                try:
                    # Check both keyboard input and GPIO queue
                    # Set up select to watch stdin
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)  # 0.1s timeout
                    
                    cmd = None
                    
                    # Check for keyboard input
                    if rlist:
                        cmd = sys.stdin.readline().strip().lower()
                    
                    # Check for GPIO input
                    try:
                        while not input_queue.empty():
                            cmd = input_queue.get_nowait()
                    except queue.Empty:
                        pass
                        
                    if not cmd:
                        continue

                    print(f"cmd: {cmd}")

                    # Process command
                    if cmd == 'exit' or cmd == '\x03':  # exit or Ctrl+C
                        break
                    elif cmd not in ['f', 'w', 's', 'c'] and not (cmd.isdigit() and 1 <= int(cmd) <= 4):
                        print("Invalid command")
                        continue

                    # Handle channel change
                    if cmd == 'c':
                        player.state = PlayerState.CHANGE_CHANNEL
                        print(f"Changing to next channel")
                        continue

                    # Handle file change
                    if cmd == 'f':
                        channel = player.control_channel
                        player.index_to_play_by_channel[channel] = ((player.index_to_play_by_channel[channel] + 1) 
                                                               % len(channel_play_orders[channel]))
                        current_file = channel_play_orders[channel][player.index_to_play_by_channel[channel]]
                        print(f"Channel {channel + 1} now playing file {current_file}")
                        print("Playing next file announcement...")
                        player.state = PlayerState.PLAY_NEXT_FILE_ANNOUNCEMENT
                    elif cmd == 'w':
                        print("Checking for win...")
                        files_to_play_by_channel = [channel_play_orders[c][player.index_to_play_by_channel[c]] for c in range(CHANNELS)]
                        files_to_play_by_channel = [x for x in files_to_play_by_channel if x != 0]
                        max_dupe_count = max_duplicate_count(files_to_play_by_channel)

                        if max_dupe_count == CHANNELS:
                            print("Winner found!")
                            player.state = PlayerState.PLAY_VICTORY_ANNOUNCEMENT
                            player.winning_file = files_to_play_by_channel[0]                  
                        else:
                            print(f"No win - {max_dupe_count} matches")
                            player.duplicate_count = max_dupe_count
                            player.state = PlayerState.PLAY_MATCH_ANNOUNCEMENT
                    elif cmd == 's':
                        for file in player.files:
                            file.current_frame = 0
                        print("Restarted all files from beginning")
                    elif cmd.isdigit():
                        channel = int(cmd) - 1
                        if 0 <= channel < CHANNELS:
                            player.control_channel = channel
                            player.channel_announce_file = player.channel_announce_files[channel]
                            player.channel_announce_file.current_frame = 0
                            player.state = PlayerState.PLAYING_CHANNEL_ANNOUNCEMENT
                            player.announcement_start_time = time.time()
                            print(f"Switched to channel {channel + 1}")

                    # Check if all songs have been matched
                    if player.should_exit:
                        print("All songs matched - exiting!")
                        break

                except KeyboardInterrupt:
                    break

    finally:
        if IS_RASPBERRY_PI and use_gpio:
            try:
                GPIO.cleanup()
            except Exception as e:
                print(f"Error cleaning up GPIO: {e}")
    print("Done!")

if __name__ == "__main__":
    main() 