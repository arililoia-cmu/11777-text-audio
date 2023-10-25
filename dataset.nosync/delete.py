import os
import sys
import csv


def delete_audio(name):
    audio_path = os.path.join('audio/', name + '.m4a')
    try:
        os.remove(audio_path)
        return True
    except:
        return False
    

if __name__ == '__main__':
    start, end = int(sys.argv[1]), int(sys.argv[2])
    with open('song_tags.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        rows = list(reader)[start:end]
    
    failure = 0
    for row in rows:
        sucess = delete_audio(row[1])
        if not sucess:
            failure += 1
    print(f'Failed to delete {failure} files')