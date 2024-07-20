from pydub import AudioSegment

def convert_audio_to_wav(input_file, output_file):
    audio = AudioSegment.from_file(input_file)
    audio.export(output_file, format="wav")


input_filename = './audio/audio_2024-07-20_15-53-29.ogg'
output_filename = './audio/audio_2024-07-20_15-53-29.wav'
convert_audio_to_wav(input_filename, output_filename)