import csv
import os
import sys
import time
import librosa
import numpy as np
import json
from tqdm import tqdm
from collections import defaultdict, OrderedDict

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

def check_exist():
    with open('song_tags.json', 'r') as f:
        song_tags = json.load(f)
    
    # Check if corresponding .npy file exists
    not_exit = []
    for msd_id in tqdm(song_tags):
        path = os.path.join('npy/', msd_id + '.npy')
        if not os.path.exists(path):
            print(f'{msd_id} does not exist')
            not_exit.append(msd_id)
    for msd_id in not_exit:
        del song_tags[msd_id]
    json.dump(song_tags, open('song_tags.json', 'w'))

def check_length():
    with open('song_tags.json', 'r') as f:
        song_tags = json.load(f)
    
    short = []
    for msd_id in tqdm(song_tags):
        path = os.path.join('npy/', msd_id + '.npy')
        audio = np.load(path, mmap_mode='r')
        if len(audio) < 16000 * 10:
            short.append(msd_id)
            os.remove(path)
    
    print(f'Number of short songs: {len(short)}')

    for msd_id in short:
        del song_tags[msd_id]
    json.dump(song_tags, open('song_tags.json', 'w'))

def match_split():
    with open('song_tags.json', 'r') as f:
        song_tags = json.load(f)
    
    with open('split.json', 'r') as f:
        split = json.load(f)
    
    total = 0
    for key in split:
        delete = []
        for msd_id in split[key]:
            if msd_id not in song_tags:
                delete.append(msd_id)
        total += len(delete)
        for msd_id in delete:
            split[key].remove(msd_id)
    json.dump(split, open('split.json', 'w'))
    print(f'Total: {total}')

def split_tag():
    with open('split.json', 'r') as f:
        split = json.load(f)
    
    with open('song_tags.json', 'r') as f:
        song_tags = json.load(f)
    
    tags = dict()
    for key in split:
        key_tag = set()
        for msd_id in split[key]:
            key_tag.update(song_tags[msd_id])
        tags[key] = list(key_tag)
    json.dump(tags, open('split_tags.json', 'w'), indent=4)

def tag_stat():
    with open('song_tags.json', 'r') as f:
        song_tags = json.load(f)
    
    with open('split.json', 'r') as f:
        split = json.load(f)
    
    stat = defaultdict(int)
    for msd_id in song_tags:
        for tag in song_tags[msd_id]:
            stat[tag] += 1
    stat = OrderedDict(sorted(stat.items(), key=lambda x: x[1], reverse=True))
    json.dump(stat, open('tag_stat.json', 'w'), indent=4)


if __name__ == '__main__':
    # convert_audio()
    # split_data()
    # get_song_tags()
    # check_exist()
    # check_length()
    # match_split()
    # tag_stat()
    # split_tag()


    with open('split.json', 'r') as f:
        split = json.load(f)
    print(f'Train: {len(split["train_track"]) + len(split["extra_track"])}')
    print(f'Valid: {len(split["valid_track"])}')
    print(f'Test: {len(split["test_track"])}')
    with open('split_tags.json', 'r') as f:
        split_tags = json.load(f)
    print(f'Train tags: {len(split_tags["train_track"])}')
    print(f'Valid tags: {len(split_tags["valid_track"])}')
    print(f'Test tags: {len(split_tags["test_track"])}')

    

