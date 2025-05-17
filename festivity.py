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
import termios
import tty
import select
import curses
import time
import platform

# GPIO Configuration - only import and use on Raspberry Pi
IS_RASPBERRY_PI = platform.system() == 'Linux' and platform.machine().startswith('arm')
if IS_RASPBERRY_PI:
    try:
        import RPi.GPIO as GPIO
        # Using BCM numbering (GPIO numbers)
        CHANNEL_BUTTON_PIN = 17  # GPIO17 - for channel control
        WIN_BUTTON_PIN = 27      # GPIO27 - for win check
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
        GPIO.setup(CHANNEL_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(WIN_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        print(f"GPIO setup successful on GPIO{CHANNEL_BUTTON_PIN} and GPIO{WIN_BUTTON_PIN}")
        return True
    except Exception as e:
        print(f"Error setting up GPIO: {e}")
        return False

class PlayerState(Enum):
    IDLE = auto()
    PLAYING_CHANNEL_ANNOUNCEMENT = auto()
    PLAY_MATCH_ANNOUNCEMENT = auto()
    PLAYING_MATCH_ANNOUNCEMENT = auto()
    PLAY_VICTORY_ANNOUNCEMENT = auto()
    PLAYING_VICTORY_ANNOUNCEMENT = auto()
    PLAY_VICTORY_FILE = auto()
    PLAYING_VICTORY_FILE = auto()
    PLAYED_VICTORY_FILE = auto()

MUSIC_DIR = 'music'
VOICE_DIR = 'voices'
CHANNELS = 4
SONG_FILES = [
    '0.wav',
    '1.wav',
    '2.wav',
    '3.wav'
]
FULL_SONG_FILES = [
    '0.full.wav',
    '1.full.wav',
    '2.full.wav',
    '3.full.wav'
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
FILE_COUNT = len(SONG_FILES)
CHANNEL_VOLUME = [2, 1, 0.2, 0.1]
@dataclass
class AudioFile:
    data: np.ndarray
    samplerate: int
    current_frame: int = 0

# Generate random play orders for each channel
channel_play_orders = [
    np.random.permutation(len(SONG_FILES)+1).tolist() for _ in range(CHANNELS)
]
print(f"channel_play_orders {channel_play_orders}")

class AudioPlayer:
    def __init__(self):
        self.files = []
        self.full_files = []  # Store full song files
        # Initialize indices to play file 4 for each channel by finding its position in each sequence
        self.index_to_play_by_channel = [
            channel_play_orders[channel].index(FILE_COUNT)
            for channel in range(CHANNELS)
        ]
        print(f"index_to_play_by_channel {self.index_to_play_by_channel}")
        self.control_channel = 0
        self.duplicate_count = None
        self.winning_file = None  # Track which file has won but not finished playing
        self.done_playing_announcement = False  # Track if current announcement is done playing
        self.state = PlayerState.IDLE
        self.playing_victory_announcement = False
        
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
            usb_device = None
            for i, dev in enumerate(devices):
                if 'USB Audio' in dev['name']:
                    usb_device = i
                    break
            
            if usb_device is None:
                print("\nUSB Audio device not found, using default device")
                device_id = None
            else:
                print(f"\nUsing USB Audio device {usb_device}")
                device_id = usb_device
            
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
        
        # Use all available output channels
        self.channels = self.max_output_channels
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

    def _process_song_chunks(self, frames: int, should_loop: bool = True) -> List[np.ndarray]:
        """Process audio chunks for all files, handling looping and channel selection.
        
        During normal play, uses fragment files. During victory, uses full song for winning file.
        
        Args:
            frames: Number of frames to process
            should_loop: Whether files should loop when they reach the end
            
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
                if should_loop:
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
        # if state != PlayerState.IDLE:
        #     print(f"current state: {state}")
        if state == PlayerState.PLAYING_CHANNEL_ANNOUNCEMENT:
            return PlayerState.IDLE
        elif state == PlayerState.PLAYING_MATCH_ANNOUNCEMENT:
            return PlayerState.IDLE
        elif state == PlayerState.PLAYING_VICTORY_ANNOUNCEMENT:
            return PlayerState.PLAY_VICTORY_FILE
        elif state == PlayerState.PLAYING_VICTORY_FILE:
            return PlayerState.PLAYED_VICTORY_FILE
        elif state == PlayerState.PLAYED_VICTORY_FILE:
            return PlayerState.IDLE
        return PlayerState.IDLE
        # print(f"current state: {state}")

    def callback(self, outdata, frames, time_info, status):
        control_channel = int(time_info.outputBufferDacTime) % CHANNELS
        if self.state == PlayerState.IDLE:
            if control_channel != self.control_channel:
                self.control_channel = control_channel
                self.channel_announce_file = self.channel_announce_files[control_channel]
                self.channel_announce_file.current_frame = 0
                self.state = PlayerState.PLAYING_CHANNEL_ANNOUNCEMENT
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

        chunks = self._process_song_chunks(frames, not self.is_victory_state)
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
            # Try announcements on channels 6 and 7 (indices 5 and 6)
            if mixed.shape[1] > 6:
                mixed[:, 5] += channel_announce_chunk * 0.5  # Channel 6
                mixed[:, 6] += channel_announce_chunk * 0.5  # Channel 7
            # Fallback to channels 4-5 if 6-7 not available
            elif mixed.shape[1] > 4:
                mixed[:, 4] += channel_announce_chunk * 0.5  # Channel 5
                mixed[:, 5] += channel_announce_chunk * 0.5  # Channel 6
            # Fallback to channels 0-1 if no other channels available
            else:
                mixed[:, 0] += channel_announce_chunk * 0.5  # Channel 1
                mixed[:, 1] += channel_announce_chunk * 0.5  # Channel 2

        # Copy announcement to first four channels during victory states
        if self.state in [PlayerState.PLAYING_VICTORY_ANNOUNCEMENT]:
            for i in range(min(4, mixed.shape[1])):  # copy to first 4 channels
                mixed[:, i] = channel_announce_chunk

        mixed = self._normalize_volume(mixed)
        outdata[:] = mixed

    def _handle_win(self, winning_file: int) -> None:
        """Remove the winning file from all play orders and reset channels to silent track."""
        # return
        for i in range(len(channel_play_orders)):
            channel_play_orders[i] = [x for x in channel_play_orders[i] if x != winning_file]
            # Reset to silent track
            self.index_to_play_by_channel = [
                channel_play_orders[channel].index(FILE_COUNT) for channel in range(CHANNELS)
            ]
        # print(f"Removed file {winning_file} from play orders")
        # print(f"channel_play_orders: {channel_play_orders}")
        # print(f"Updated channel_play_orders: {channel_play_orders}")
        # print(f"Updated index_to_play_by_channel: {self.index_to_play_by_channel}")

class TerminalUI:
    def __init__(self):
        self.stdscr = curses.initscr()
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)  # For status messages
        curses.init_pair(2, curses.COLOR_YELLOW, -1)  # For channel info
        curses.init_pair(3, curses.COLOR_RED, -1)    # For warnings/errors
        curses.init_pair(4, curses.COLOR_CYAN, -1)   # For victory states
        
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)
        self.stdscr.clear()
        self.stdscr.refresh()
        
        # Create windows
        height, width = self.stdscr.getmaxyx()
        self.status_win = curses.newwin(3, width, 0, 0)
        self.channel_win = curses.newwin(CHANNELS + 2, width, 3, 0)
        self.info_win = curses.newwin(5, width, CHANNELS + 5, 0)
        
        # Enable scrolling for info window
        self.info_win.scrollok(True)
        self.info_win.idlok(True)
        
    def cleanup(self):
        curses.nocbreak()
        self.stdscr.keypad(False)
        curses.echo()
        curses.endwin()
        
    def update_status(self, message, color_pair=1):
        self.status_win.clear()
        self.status_win.addstr(1, 1, message, curses.color_pair(color_pair))
        self.status_win.refresh()
        
    def update_channels(self, player):
        self.channel_win.clear()
        self.channel_win.addstr(0, 1, "Channel Status:", curses.A_BOLD)
        
        for channel in range(CHANNELS):
            current_file = channel_play_orders[channel][player.index_to_play_by_channel[channel]]
            status = "Playing" if current_file != FILE_COUNT else "Silent"
            file_name = SONG_FILES[current_file] if current_file != FILE_COUNT else "---"
            color = curses.color_pair(2)
            if channel == player.control_channel:
                color |= curses.A_BOLD
            self.channel_win.addstr(channel + 1, 1, 
                f"Channel {channel + 1}: {status} - {file_name}", color)
                # f"Channel {channel + 1}: {status} - {file_name}", color)
                
        # Show remaining frames during victory playback
        if player.is_victory_state:
            remaining_frames = len(player.full_files[player.winning_file].data) - player.full_files[player.winning_file].current_frame
            remaining_seconds = remaining_frames / player.samplerate
            self.channel_win.addstr(CHANNELS + 1, 1, 
                f"Remaining: {remaining_seconds:.1f}s ({remaining_frames} frames)", 
                curses.color_pair(4))
                
        self.channel_win.refresh()
        
    def add_info(self, message, color_pair=1):
        self.info_win.addstr(f"{message}\n", curses.color_pair(color_pair))
        self.info_win.refresh()
        
    def clear_info(self):
        self.info_win.clear()
        self.info_win.refresh()

class DebugUI:
    def __init__(self):
        print("\n=== Debug UI Mode ===")
        print("Controls:")
        print("1-4: Select channel")
        print("Space: Control current channel")
        print("Enter: Check for win")
        print("S: Restart all files")
        print("Ctrl+C: Exit")
        print("===================\n")
        
    def cleanup(self):
        print("\nCleaning up...")
        
    def update_status(self, message, color_pair=1):
        print(f"\nStatus: {message}")
        
    def update_channels(self, player):
        print("\nChannel Status:")
        for channel in range(CHANNELS):
            current_file = channel_play_orders[channel][player.index_to_play_by_channel[channel]]
            status = "Playing" if current_file != FILE_COUNT else "Silent"
            file_name = SONG_FILES[current_file] if current_file != FILE_COUNT else "---"
            print(f"Channel {channel + 1}: {status} - {file_name}")
            
        # Show remaining frames during victory playback
        if player.is_victory_state:
            remaining_frames = len(player.full_files[player.winning_file].data) - player.full_files[player.winning_file].current_frame
            remaining_seconds = remaining_frames / player.samplerate
            print(f"\nRemaining: {remaining_seconds:.1f}s ({remaining_frames} frames)")
        
    def add_info(self, message, color_pair=1):
        print(f"Info: {message}")
        
    def clear_info(self):
        pass

def list_audio_devices():
    print("\nAvailable audio devices:")
    print(sd.query_devices())

def max_duplicate_count(nums):
    counts = Counter(nums)
    duplicate_counts = [count for count in counts.values() if count > 1]
    return max(duplicate_counts) if duplicate_counts else 0
    
def get_single_key():
    """Read a single keypress from stdin without requiring Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        # Use select to check if there's input available
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            ch = sys.stdin.read(1)
        else:
            ch = None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def main():
    # Check if we're in a terminal
    is_tty = sys.stdout.isatty()
    
    if is_tty:
        try:
            ui = TerminalUI()
        except curses.error:
            print("Warning: Could not initialize curses UI, falling back to debug mode")
            ui = DebugUI()
    else:
        ui = DebugUI()
    
    try:
        # Initialize GPIO first if on Raspberry Pi
        use_gpio = False
        if IS_RASPBERRY_PI:
            if setup_gpio():
                use_gpio = True
                ui.add_info("GPIO setup successful", 1)
            else:
                ui.add_info("Failed to initialize GPIO. Running without button support.", 3)
        else:
            ui.add_info("Running without GPIO support (not on Raspberry Pi)", 2)
            
        try:
            player = AudioPlayer()
        except Exception as e:
            ui.add_info(f"Error initializing audio: {str(e)}", 3)
            ui.cleanup()
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
            while True:
                try:
                    # Read current states
                    channel_state = GPIO.input(CHANNEL_BUTTON_PIN)
                    win_state = GPIO.input(WIN_BUTTON_PIN)
                    current_time = time.time()
                    
                    # Check for channel button press (FALLING edge)
                    if channel_state == GPIO.LOW and last_channel_state == GPIO.HIGH:
                        if current_time - last_channel_button_press > DEBOUNCE_TIME:
                            input_queue.put('space')
                            last_channel_button_press = current_time
                            ui.add_info("Channel button pressed", 2)
                    
                    # Check for win button press (FALLING edge)
                    if win_state == GPIO.LOW and last_win_state == GPIO.HIGH:
                        if current_time - last_win_button_press > DEBOUNCE_TIME:
                            input_queue.put('enter')
                            last_win_button_press = current_time
                            ui.add_info("Win button pressed", 2)
                    
                    # Update last states
                    last_channel_state = channel_state
                    last_win_state = win_state
                    
                    # Small sleep to prevent CPU hogging
                    time.sleep(0.1)
                except Exception as e:
                    ui.add_info(f"Error in GPIO poll thread: {e}", 3)
                    break
        
        # Start GPIO poll thread if on Raspberry Pi and GPIO is available
        if use_gpio:
            gpio_thread = threading.Thread(target=gpio_poll_thread, daemon=True)
            gpio_thread.start()
            ui.add_info("GPIO polling started", 1)
        
        def input_thread():
            """Thread to read keyboard input and put it in the queue."""
            while True:
                try:
                    key = get_single_key()
                    if key is None:
                        continue
                    if key == '\x0d' or key == '\n':  # Enter key
                        input_queue.put('enter')
                    elif key == ' ':  # Space key
                        input_queue.put('space')
                    elif key == 's':
                        input_queue.put('s')
                    elif key.isdigit():
                        input_queue.put(key)
                    elif key == '\x03':  # Ctrl+C
                        raise KeyboardInterrupt
                except KeyboardInterrupt:
                    break

        # Start input thread
        thread = threading.Thread(target=input_thread, daemon=True)
        thread.start()

        # Start audio playback
        with sd.OutputStream(
            samplerate=player.samplerate,
            channels=player.channels,
            dtype='float32',
            callback=player.callback
        ):
            ui.update_status("Playing audio... Press Ctrl+C to exit", 1)
            last_state = None
            
            while True:
                try:
                    # Update UI if state changed
                    if player.state != last_state:
                        if player.is_victory_state:
                            ui.update_status(f"Playing victory file {player.winning_file}", 4)
                        elif player.state == PlayerState.PLAYING_CHANNEL_ANNOUNCEMENT:
                            ui.update_status(f"Playing channel {player.control_channel + 1} announcement", 2)
                        elif player.state == PlayerState.PLAYING_MATCH_ANNOUNCEMENT:
                            ui.update_status(f"Playing match announcement for {player.duplicate_count} matches", 2)
                        elif player.state == PlayerState.PLAYING_VICTORY_ANNOUNCEMENT:
                            ui.update_status("Playing victory announcement!", 4)
                        else:
                            ui.update_status("Playing audio... Press Ctrl+C to exit", 1)
                        last_state = player.state
                    
                    # Update channel display
                    ui.update_channels(player)
                    
                    # Check for input with a timeout
                    try:
                        key = input_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    # Ignore all controls while winning file is playing
                    if player.is_victory_state:
                        continue

                    channel = None
                    if key == 'space':
                        channel = player.control_channel
                        ui.add_info(f"Channel control triggered - controlling channel {channel + 1}", 2)
                    elif key == 'enter':
                        ui.add_info("Checking for win...", 1)
                        files_to_play_by_channel = [channel_play_orders[c][player.index_to_play_by_channel[c]] for c in range(CHANNELS)]
                        files_to_play_by_channel = [x for x in files_to_play_by_channel if x != FILE_COUNT]
                        max_dupe_count = max_duplicate_count(files_to_play_by_channel)

                        if max_dupe_count == FILE_COUNT:
                            ui.add_info("Winner found!", 4)
                            player.state = PlayerState.PLAY_VICTORY_ANNOUNCEMENT
                            player.winning_file = files_to_play_by_channel[0]                  
                        else:
                            ui.add_info(f"No win - {max_dupe_count} matches", 2)
                            player.duplicate_count = max_dupe_count
                            player.state = PlayerState.PLAY_MATCH_ANNOUNCEMENT
                    elif key == 's':
                        for file in player.files:
                            file.current_frame = 0
                        ui.add_info("Restarted all files from beginning", 1)
                        continue
                    elif key.isdigit():
                        channel = int(key) - 1
                        if channel < 0 or channel >= CHANNELS:
                            continue
                        ui.add_info(f"Channel {channel + 1} selected", 2)
                    
                    if channel is not None:
                        player.index_to_play_by_channel[channel] = ((player.index_to_play_by_channel[channel] + 1) 
                                                                   % len(channel_play_orders[channel]))
                        current_files = [channel_play_orders[c][player.index_to_play_by_channel[c]] for c in range(CHANNELS)]
                        ui.add_info(f"Channel {channel + 1} now playing file {current_files[channel]}", 2)

                except KeyboardInterrupt:
                    break

    finally:
        if IS_RASPBERRY_PI and use_gpio:
            try:
                GPIO.cleanup()
            except Exception as e:
                print(f"Error cleaning up GPIO: {e}")
        ui.cleanup()
    print("Done!")

if __name__ == "__main__":
    main() 