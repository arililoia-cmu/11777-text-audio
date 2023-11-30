"""
Create subset of the dataset
"""
import json
import random
from collections import defaultdict, OrderedDict

def subset_split(sizes):
    train_size, val_size, test_size = sizes

    with open('full/split.json', 'r') as f:
        split = json.load(f)
        train_track = split['train_track']
        valid_track = split['valid_track']
        test_track = split['test_track']

        if train_size > len(train_track):
            train_track += split['extra_track']
    
    train_track = random.sample(train_track, train_size)
    valid_track = random.sample(valid_track, val_size)
    test_track = random.sample(test_track, test_size)
    sub_split = {
        'train_track': train_track,
        'valid_track': valid_track,
        'test_track': test_track
    }
    json.dump(sub_split, open('small/split.json', 'w'))
    print(f"Train: {len(train_track)}, Valid: {len(valid_track)}, Test: {len(test_track)}")
    
    tracks = train_track + valid_track + test_track
    files = [t + '.npy' for t in tracks]
    with open('small/tracklist.txt', 'w') as f:
        f.write('\n'.join(files) + '\n')

def subset_tags():
    with open('small/split.json', 'r') as f:
        split = json.load(f)
        song_ids = split['train_track'] + split['valid_track'] + split['test_track']
    
    with open('full/clustered_song_tags.json', 'r') as f:
        song_tags = json.load(f)
        sub_song_tags = {k: song_tags[k] for k in song_ids}
    json.dump(sub_song_tags, open('small/clustered_song_tags.json', 'w'))

    cluster = defaultdict(set)
    for song in sub_song_tags:
        for c in sub_song_tags[song]:
            cluster[c] |= set(sub_song_tags[song][c])
    for c in cluster:
        cluster[c] = list(cluster[c])
    json.dump(cluster, open('small/clustered_tags.json', 'w'))

    mix_song_tags = dict()
    for song in sub_song_tags:
        mix_song_tags[song] = set()
        for c in sub_song_tags[song]:
            mix_song_tags[song].update(sub_song_tags[song][c])
        mix_song_tags[song] = list(mix_song_tags[song])
    json.dump(mix_song_tags, open('small/song_tags.json', 'w'))
    
    tags = set()
    for song in mix_song_tags:
        tags.update(mix_song_tags[song])
    tags = list(tags)
    json.dump(tags, open('small/all_tags.json', 'w'), indent=4)

def subset_tag_split():
    with open('small/split.json', 'r') as f:
        split = json.load(f)
        
    with open('small/clustered_song_tags.json', 'r') as f:
        song_tags = json.load(f)
    
    clustered_split_tags = dict()
    split_tags = dict()
    for s in split:
        split_tags[s] = set()
        clustered_split_tags[s] = defaultdict(set)
        for song in split[s]:
            for c in song_tags[song]:
                split_tags[s].update(song_tags[song][c])
                clustered_split_tags[s][c] |= set(song_tags[song][c])
        split_tags[s] = list(split_tags[s])
        for c in clustered_split_tags[s]:
            clustered_split_tags[s][c] = list(clustered_split_tags[s][c])
    json.dump(split_tags, open('small/split_tags.json', 'w'))
    json.dump(clustered_split_tags, open('small/clustered_split_tags.json', 'w'))


def subset_stat():
    with open('small/clustered_song_tags.json', 'r') as f:
        song_tags = json.load(f)
    
    cluster_count = defaultdict(int)
    tag_count = defaultdict(int)
    for song in song_tags:
        for c in song_tags[song]:
            for tag in song_tags[song][c]:
                tag_count[tag] += 1
            cluster_count[c] += len(song_tags[song][c])
    tag_count = OrderedDict(sorted(tag_count.items(), key=lambda x: x[1], reverse=True))
    cluster_count = OrderedDict(sorted(cluster_count.items(), key=lambda x: x[1], reverse=True))
    json.dump([cluster_count, tag_count], open('small/cluster_stat.json', 'w'), indent=4)



if __name__ == '__main__':
    # subset_split([40000, 5000, 5000])
    subset_tags()
    subset_tag_split()
    subset_stat()