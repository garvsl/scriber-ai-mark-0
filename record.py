import sounddevice as sd
from scipy.io.wavfile import write
import threading
from pynput import keyboard
import numpy as np

recording = False
audio_chunks = []
fs = 44100  
filename = './audio/output.wav'
block_size = 1024  


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status)
    if recording:
        audio_chunks.append(indata.copy())

def record_audio():
    global recording
    recording = True
    print("Recording...")
    with sd.InputStream(samplerate=fs, blocksize=block_size, channels=1, callback=callback):
        while recording:
            sd.sleep(1000)
    print("Recording stopped")
    save_recording()

def save_recording():
    audio_array = np.concatenate(audio_chunks, axis=0)
    write(filename, fs, audio_array)
    print(f"Recording saved to {filename}")

def stop_recording():
    global recording
    recording = False
    print("Stopping recording...")


def on_press(key):
    if key == keyboard.Key.esc:
        stop_recording()
        return False 


recording_thread = threading.Thread(target=record_audio)
recording_thread.start()

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()

recording_thread.join()



