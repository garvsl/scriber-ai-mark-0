import os
import sys
from tempfile import TemporaryDirectory
from faster_whisper import WhisperModel
from subprocess import check_output
from pydub import AudioSegment

def ms_to_str(milliseconds: float, include_ms=False):
    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    formatted = f'{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}'
    if include_ms:
        formatted += f'.{int(milliseconds):03d}'
    return formatted

def convert_webm_to_wav(webm_path, wav_path):
    audio = AudioSegment.from_file(webm_path, format="webm")
    audio.export(wav_path, format="wav")

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


if __name__ == "main":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_webm_file>")
        sys.exit(1)
    
    webm_file = sys.argv[1]
    base_filename = os.path.splitext(os.path.basename(webm_file))[0]

    with TemporaryDirectory() as temp_dir:
        wav_file = os.path.join(temp_dir, f"{base_filename}.wav")
        convert_webm_to_wav(webm_file, wav_file)

        segments, info = transcribe_audio(wav_file)
        
        save_location = os.path.join(os.getcwd(), "..", "store")
        # os.makedirs('./transcripts', exist_ok=True)
        output_filepath = f'{base_filename}.txt'

        with open(output_filepath, 'w') as f:
            for segment in segments:
                text = segment.text
                f.write(f"{text}\n")

        print(f"Transcription saved to {output_filepath}")