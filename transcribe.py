import os
from tempfile import TemporaryDirectory
from faster_whisper import WhisperModel
from subprocess import  check_output


def ms_to_str(milliseconds: float, include_ms=False):
    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    formatted = f'{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}'
    if include_ms:
        formatted += f'.{int(milliseconds):03d}'
    return formatted 

def transcribe_audio(filename):
    model = WhisperModel("./models/faster-whisper-small",
            device='auto',  
            cpu_threads=int(check_output(["sysctl", "-n", "hw.perflevel0.logicalcpu_max"])), 
            compute_type="auto", 
            local_files_only=True)
    
    whisper_lang = "en"
    vad_threshold = 0.5

    segments, info = model.transcribe(
    filename, language=whisper_lang, 
    beam_size=1, temperature=0, word_timestamps=True, 
    initial_prompt= "Hmm, let me think like, hmm, okay, here's what I'm, like, thinking.", vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=200, 
                        threshold=vad_threshold))
                    
    return segments, info


filename = "./audio/audio_2024-07-20_15-53-29.wav"

segments, info = transcribe_audio(filename)

base_filename = os.path.splitext(os.path.basename(filename))[0]

os.makedirs('./transcripts', exist_ok=True)

output_filepath = f'./transcripts/{base_filename}.txt'

with open(output_filepath, 'w') as f:
    for segment in segments:
        start_time = ms_to_str(segment.start * 1000.0, include_ms=True)
        end_time = ms_to_str(segment.end * 1000.0, include_ms=True)
        text = segment.text
        f.write(f"{text}\n")
