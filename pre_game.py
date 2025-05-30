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
import select

# GPIO Configuration - only import and use on Raspberry Pi
IS_RASPBERRY_PI = platform.system() == 'Linux' and platform.machine().startswith('aarch')
if IS_RASPBERRY_PI:
    try:
        import RPi.GPIO as GPIO
        # Using BCM numbering (GPIO numbers)
        GPIO_PINS = [17, 27, 22]  # GPIO pins to monitor
        DEBOUNCE_TIME = 0.2      # Button debounce time in seconds
    except ImportError:
        print("Warning: RPi.GPIO not available. Running without GPIO support.")
        IS_RASPBERRY_PI = False

def setup_gpio():
    """Initialize GPIO pins if on Raspberry Pi."""
    if not IS_RASPBERRY_PI:
        return False
        
    try:
        GPIO.setmode(GPIO.BCM)  # Use BCM numbering
        for pin in GPIO_PINS:
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        print(f"GPIO setup successful on pins: {', '.join(map(str, GPIO_PINS))}")
        return True
    except Exception as e:
        print(f"Error setting up GPIO: {e}")
        return False

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
    
    # Flag to signal threads to exit
    exit_flag = threading.Event()

    def gpio_poll_thread():
        """Thread to poll GPIO buttons on Raspberry Pi."""
        if not IS_RASPBERRY_PI:
            return

        last_states = {pin: GPIO.HIGH for pin in GPIO_PINS}
        last_button_press = 0

        while not exit_flag.is_set():
            try:
                current_time = time.time()
                
                # Check all pins
                for pin in GPIO_PINS:
                    current_state = GPIO.input(pin)
                    
                    # Check for button press (FALLING edge)
                    if current_state == GPIO.LOW and last_states[pin] == GPIO.HIGH:
                        if current_time - last_button_press > DEBOUNCE_TIME:
                            input_queue.put('exit')
                            last_button_press = current_time
                    
                    last_states[pin] = current_state

                time.sleep(0.1)
            except Exception as e:
                print(f"Error in GPIO poll thread: {e}")
                break

    # Start the GPIO thread if available
    gpio_thread = None
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
            
            # Wait for input using select instead of input()
            while True:
                try:
                    # Check both keyboard input and GPIO queue
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                    
                    if rlist:  # If keyboard input is available
                        if sys.stdin.readline().strip():  # Read and clear the input
                            break
                            
                    # Check GPIO queue
                    try:
                        cmd = input_queue.get_nowait()
                        if cmd == 'exit':
                            break
                    except queue.Empty:
                        pass
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error in main loop: {e}")
                    break

    except Exception as e:
        print(f"Error during playback: {e}")
    finally:
        # Signal threads to exit
        exit_flag.set()
        
        # Clean up GPIO
        if IS_RASPBERRY_PI and use_gpio:
            try:
                GPIO.cleanup()
            except Exception as e:
                print(f"Error cleaning up GPIO: {e}")

    print("Done!")

if __name__ == "__main__":
    main() 