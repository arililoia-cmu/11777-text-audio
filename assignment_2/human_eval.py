import csv
import ast
import zipfile

def collect_tags(n, a, b):
    """Collect n tags from songs that have between a and b tags"""
    # Read csv file from song_tags.csv
    rows = csv.reader(open('dataset.nosync/song_tags.csv', 'r'))
    next(rows)

    tags = set()
    songs = set()
    for row in rows:
        tag = row[2]
        tag = ast.literal_eval(tag)
        if a <= len(tag) <= b:
            tags.update(tag)
            songs.add(row[1])
        if len(tags) > n:
            break
    
    tag_out = open('dataset.nosync/tags.txt', 'w')
    tag_out.write('\n'.join(tags))
    tag_out.close()
    song_out = open('dataset.nosync/songs.txt', 'w')
    song_out.write('\n'.join(songs))
    song_out.close()

    # Zip the audio files
    with zipfile.ZipFile('dataset.nosync/songs.zip', 'w') as zipf:
        for song in songs:
            zipf.write(f'dataset.nosync/audio/{song}.m4a', arcname=f'{song}.m4a')

if __name__ == '__main__':
    collect_tags(50, 3, 5)
