import os
import shlex
from tempfile import TemporaryDirectory
from faster_whisper import WhisperModel
from subprocess import PIPE, STDOUT, Popen, check_output
import yaml

def overlap_len(ss_start, ss_end, ts_start, ts_end):
  
    if ts_end < ss_start: 
        return None

    if ts_start > ss_end:
        return 0.0

    ts_len = ts_end - ts_start
    if ts_len <= 0:
        return None

    overlap_start = max(ss_start, ts_start)
    overlap_end = min(ss_end, ts_end) 

    ol_len = overlap_end - overlap_start + 1
    return ol_len / ts_len

def find_speaker(diarization, transcript_start, transcript_end) -> str:
    spkr = ''
    overlap_found = 0
    overlap_threshold = 0.8
    segment_len = 0
    is_overlapping = False

    for segment in diarization:
        t = overlap_len(segment["start"], segment["end"], transcript_start, transcript_end)
        if t is None:
            break

        current_segment_len = segment["end"] - segment["start"] 
        current_segment_spkr = f'S{segment["label"][8:]}' 

        if overlap_found >= overlap_threshold:
            if (t >= overlap_threshold) and (current_segment_len < segment_len): 
                is_overlapping = True
                overlap_found = t
                segment_len = current_segment_len
                spkr = current_segment_spkr
        elif t > overlap_found: 
            overlap_found = t
            segment_len = current_segment_len
            spkr = current_segment_spkr
        
    if is_overlapping:
        return f"//{spkr}"
    else:
        return spkr
    
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

for segment in segments:
    start = round(segment.start * 1000.0)
    end = round(segment.end * 1000.0)

    seg_text = segment.text
    seg_html = seg_text

    tmpdir = TemporaryDirectory('transcribe')
    app_dir = os.path.abspath(os.path.dirname(__file__))

    diarize_output = os.path.join(tmpdir.name, 'diarize_out.yaml')
    print(app_dir)
    diarize_abspath = 'python ' + os.path.join(app_dir, 'diarize.py')


    diarize_cmd = f'{diarize_abspath} {'cpu'} "{filename}" "{diarize_output}" {'auto'}'
    diarize_env = None


    diarize_cmd = shlex.split(diarize_cmd)
    startupinfo = None


    with Popen(diarize_cmd,
                stdout=PIPE,
                stderr=STDOUT,
                encoding='UTF-8',
                startupinfo=startupinfo,
                env=diarize_env,
                close_fds=True) as pyannote_proc:
        for line in pyannote_proc.stdout:
            print(line)
            if line.startswith('progress '):
                progress = line.split()
                step_name = progress[1]
                progress_percent = int(progress[2])
                                      
                if step_name == 'segmentation':
                    print(2, progress_percent * 0.3)
                elif step_name == 'embeddings':
                    print(2, 30 + (progress_percent * 0.7))
            elif line.startswith('error '):
                print('error')
            elif line.startswith('log: '):
                print('log')
                if line.strip() == "log: 'pyannote_xpu: cpu' was set.": 
                    print('dunno')

    if pyannote_proc.returncode > 0:
        raise Exception('')

    with open(diarize_output, 'r') as file:
        diarization = yaml.safe_load(file)

    for segment in diarization:
        line = f'{ms_to_str(segment["start"], include_ms=True)} - {ms_to_str(segment["end"], include_ms=True)} {segment["label"]}'
        print(line)
    