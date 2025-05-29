#!/usr/bin/env python3

"""A simple program that plays a dialtone on all 6 tracks of a USB audio device.
Exits when Enter is pressed or when GPIO buttons (17 or 27) are pressed.
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import sys
import threading
import queue
import time
import platform

# GPIO Configuration - only import and use on Raspberry Pi
IS_RASPBERRY_PI = platform.system() == 'Linux' and platform.machine().startswith('aarch')
if IS_RASPBERRY_PI:
    try:
        import RPi.GPIO as GPIO
        # Using BCM numbering (GPIO numbers)
        CHANNEL_BUTTON_PIN = 17  # GPIO17
        WIN_BUTTON_PIN = 27      # GPIO27
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

def get_input():
    """Read input from stdin."""
    try:
        return input()
    except EOFError:
        return None

def audio_callback(outdata, frames, time_info, status):
    """Callback function for audio playback."""
    global current_frame
    
    if status:
        print(status)
    
    # Get the next chunk of audio data
    chunk = audio_data[current_frame:current_frame + frames]
    
    # If we don't have enough frames, loop back to start
    if len(chunk) < frames:
        # Calculate how many frames we still need
        remaining_frames = frames - len(chunk)
        # Get the remaining frames from the start of the audio
        extra_chunk = audio_data[0:remaining_frames]
        # Concatenate the chunks
        chunk = np.concatenate([chunk, extra_chunk])
        # Reset current_frame to continue from where we left off
        current_frame = remaining_frames
    else:
        # Advance the frame counter
        current_frame += frames
        # Loop back if we've reached the end
        if current_frame >= len(audio_data):
            current_frame = 0
    
    # Reshape to match output channels (5 active + 1 silent)
    chunk = chunk.reshape(-1, 1)  # Make it a column vector
    chunk = np.tile(chunk, (1, 5))  # Copy to first 5 channels
    # Add a silent 6th channel
    chunk = np.pad(chunk, ((0, 0), (0, 1)), mode='constant', constant_values=0)
    
    # Copy the data to the output buffer
    outdata[:] = chunk

def main():
    global audio_data, current_frame
    
    # Load the dialtone audio file
    try:
        audio_data, samplerate = sf.read('voices/dialtone.wav')
        current_frame = 0
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    # Initialize GPIO if on Raspberry Pi
    use_gpio = False
    if IS_RASPBERRY_PI:
        use_gpio = setup_gpio()
        if use_gpio:
            print("GPIO initialized successfully")
        else:
            print("Failed to initialize GPIO. Running without button support.")

    # Create a queue for input events
    input_queue = queue.Queue()

    def gpio_poll_thread():
        """Thread to poll GPIO buttons on Raspberry Pi."""
        if not IS_RASPBERRY_PI:
            return

        last_channel_state = GPIO.HIGH
        last_win_state = GPIO.HIGH
        last_button_press = 0

        while True:
            try:
                # Read current states
                channel_state = GPIO.input(CHANNEL_BUTTON_PIN)
                win_state = GPIO.input(WIN_BUTTON_PIN)
                current_time = time.time()

                # Check for any button press (FALLING edge)
                if ((channel_state == GPIO.LOW and last_channel_state == GPIO.HIGH) or
                    (win_state == GPIO.LOW and last_win_state == GPIO.HIGH)):
                    if current_time - last_button_press > DEBOUNCE_TIME:
                        input_queue.put('exit')
                        last_button_press = current_time

                # Update last states
                last_channel_state = channel_state
                last_win_state = win_state

                time.sleep(0.1)
            except Exception as e:
                print(f"Error in GPIO poll thread: {e}")
                break

    def input_thread():
        """Thread to read keyboard input."""
        while True:
            try:
                if get_input() is not None:
                    input_queue.put('exit')
                    break
            except:
                break

    # Start the input thread
    input_thread = threading.Thread(target=input_thread, daemon=True)
    input_thread.start()

    # Start the GPIO thread if available
    if use_gpio:
        gpio_thread = threading.Thread(target=gpio_poll_thread, daemon=True)
        gpio_thread.start()

    # List available audio devices
    print("\nAvailable audio devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        print(f"{i}: {dev['name']} (in={dev['max_input_channels']}, out={dev['max_output_channels']})")

    # Try to find USB audio device
    device_id = None
    for i, dev in enumerate(devices):
        if 'USB Audio' in dev['name']:
            device_id = i
            print(f"\nUsing USB Audio device {device_id}")
            break

    if device_id is None:
        print("\nUSB Audio device not found, using default device")

    try:
        # Start audio playback
        with sd.OutputStream(
            device=device_id,
            samplerate=samplerate,
            channels=6,
            callback=audio_callback,
            dtype='float32'
        ):
            print("\nPlaying dialtone... Press Enter to exit")
            
            # Wait for input
            while True:
                try:
                    # Check for input with timeout
                    try:
                        input_queue.get(timeout=0.1)
                        break  # Exit if any input received
                    except queue.Empty:
                        continue
                except KeyboardInterrupt:
                    break

    except Exception as e:
        print(f"Error during playback: {e}")
    finally:
        if IS_RASPBERRY_PI and use_gpio:
            GPIO.cleanup()

    print("Done!")

if __name__ == "__main__":
    main() 