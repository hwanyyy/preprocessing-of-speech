import os
from os.path import join
import argparse
import numpy as np
from tqdm import tqdm

import webrtcvad
from scipy import signal
from scipy.io import wavfile
import pyloudnorm as pyln

import warnings
warnings.filterwarnings('ignore')

vad = webrtcvad.Vad()
# 1~3 까지 설정 가능, 높을수록 aggressive
vad.set_mode(3)


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    frames = []
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        frames.append(Frame(audio[offset:offset + n], timestamp, duration))
        timestamp += duration
        offset += n

    return frames


def auto_vad(vad, samples, sample_rate, frame_duration_ms):
    '''
    앞뒤에 음성이 들어가지 않는 것은 가능하나, 처음이나 끝에만 음성이 있는 경우는 불가능
    frame_duration_ms = 10 or 20 or 30 (ms)
    '''
    not_speech = []
    speech = []
    frames = frame_generator(frame_duration_ms, samples, sample_rate)
    n_frame = len(frames)

    for idx, frame in enumerate(frames):
        if not vad.is_speech(frame.bytes, sample_rate):
            not_speech.append(idx)
        else:
            speech.append(idx)

    prior = 0
    cutted_samples = []

    for i in not_speech:
        if i - prior > 2:

            start = int((float(prior) / n_frame) * len(samples))
            end = int((float(i) / n_frame) * len(samples))

            if len(cutted_samples) == 0:
                cutted_samples = samples[start:end]
            else:
                cutted_samples = np.append(cutted_samples, samples[start:end])

        prior = i

    # If there is only speech at the beginning or end, or if the speech is too short to convert, return the original sample
    if type(cutted_samples) == list:
        return samples

    return cutted_samples


def main():
    parser = argparse.ArgumentParser(description='Preprocessing of Speech')
    parser.add_argument('--opt', type=int, default=3, help='preprecessing mode (default: 3)')
    parser.add_argument('--path', type=str, default=os.getcwd(), help='wav file location (default: current directory)')

    args = parser.parse_args()
    path = args.path +'/'
    
    new_sample_rate = 8000      # Frequently related frequencies of speech exist in the lower bands
    meter = pyln.Meter(new_sample_rate)  # create BS.1770 meter

    audio_list = [x for x in os.listdir(join(path)) if x.endswith('.wav')]
    
    if args.opt == 1:
        os.mkdir(path + 'VAD_data')
        
        for i in tqdm(audio_list, ncols=100):
            sample_rate, samples = wavfile.read(str(path) + i)
            
            # VAD
            cutted_samples = auto_vad(vad, samples, sample_rate, 10)

            # loudness normalize audio to -12 dB LUFS : Set to full volume
            loudness = meter.integrated_loudness(cutted_samples)
            loudness_normalized_audio = pyln.normalize.loudness(cutted_samples, loudness, -12.0)
            
            wavfile.write(path + '/preprocessing_data/' + i, rate=sample_rate , data=loudness_normalized_audio)
            
        print('complete-!')
        
    elif args.opt == 2:
        os.mkdir(path + 'resampled_data')
        
        for i in tqdm(audio_list, ncols=100):
            sample_rate, samples = wavfile.read(str(path) + i)
            
            # resampling
            resampled = signal.resample(samples, int(new_sample_rate/sample_rate * samples.shape[0]))
            
            # loudness normalize audio to -12 dB LUFS : Set to full volume
            loudness = meter.integrated_loudness(resampled)
            loudness_normalized_audio = pyln.normalize.loudness(resampled, loudness, -12.0)
            
            wavfile.write(path + '/preprocessing_data/' + i, rate=new_sample_rate , data=loudness_normalized_audio)
            
        print('complete-!')
        
    else: 
        os.mkdir(path + 'VAD_resampled_data')
        
        for i in tqdm(audio_list, ncols=100):
            sample_rate, samples = wavfile.read(str(path) + i)
            # VAD & resampling
            cutted_samples = auto_vad(vad, samples, sample_rate, 10)
            cutted_resampled = signal.resample(cutted_samples, int(new_sample_rate/sample_rate * cutted_samples.shape[0]))

            # loudness normalize audio to -12 dB LUFS : Set to full volume
            loudness = meter.integrated_loudness(cutted_resampled)
            loudness_normalized_audio = pyln.normalize.loudness(cutted_resampled, loudness, -12.0)
            
            wavfile.write(path + '/preprocessing_data/' + i, rate=new_sample_rate , data=loudness_normalized_audio)
            
        print('complete-!')


if __name__ == "__main__":
    main()
