import csv
import os
import sys
import time
import librosa
import numpy as np
import json
from tqdm import tqdm

def load_audio(path, sr=16000, duration=30):
    try:
        y, sr = librosa.load(path, sr=sr)
        if len(y) > sr * duration:
            y = y[:sr * duration]
        return y, sr
    except:
        return None, None

def audio_to_npy(name, sr=16000):
    audio_path = os.path.join('audio/', name + '.m4a')
        
    y, sr = load_audio(audio_path, sr=sr)
    if y is None:
        return False
        
    np.save(f'npy/{name}.npy', y)
    return True

def convert_audio():
    if sys.argc == 3:
        start, end = int(sys.argv[1]), int(sys.argv[2])
    else:
        raise ValueError('Usage: python preprocess.py start end')
    
    with open('song_tags.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        rows = list(reader)[start:end]
        print(f'Processing {len(rows)} rows')
    
    sr = 16000
    begin = time.time()
    failure = 0
    for row in tqdm(rows):
        success = audio_to_npy(row[1], sr=sr)
        if not success:
            failure += 1
    print(f'Time elapsed: {(time.time() - begin) / 60:.2f} minutes')
    print(f'Failure: {failure}')

def split_data():
    with open('ECALS/ecals_annotation/ecals_track_split.json', 'r') as f:
        ecals_split = json.load(f)
    
    with open('song_tags.csv', 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        songs = set([row[1] for row in rows])
    
    new_split = dict()
    for key in ecals_split:
        new_split[key] = [song for song in ecals_split[key] if song in songs]
        print(f'{key} original: {len(ecals_split[key])}, new: {len(new_split[key])}')
    json.dump(new_split, open('split.json', 'w'))

def get_song_tags():
    with open('ECALS/ecals_annotation/annotation.json', 'r') as f:
        ecals_annotation = json.load(f)
    
    with open('song_tags.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)
    
    song_tags = dict()
    for row in rows:
        msd_id = row[1]
        song_tags[msd_id] = ecals_annotation[msd_id]['tag']
    json.dump(song_tags, open('song_tags.json', 'w'))
    print(f'Number of songs: {len(song_tags)}')


if __name__ == '__main__':
    # convert_audio()
    # split_data()
    get_song_tags()
    