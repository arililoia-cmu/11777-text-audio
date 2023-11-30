import json
from collections import defaultdict, OrderedDict

def cluster():
    with open('dataset.nosync/cluster.txt') as f:
        lines = f.readlines()
    
    lines = [l.strip().lower() for l in lines]
    i = 0
    group = dict()
    while lines[i] != '2':
        if lines[i] == '1' or lines[i] == '':
            i += 1
            continue
        group[lines[i]] = 'genre'
        i += 1
    while lines[i] != '3':
        if lines[i] == '2' or lines[i] == '':
            i += 1
            continue
        group[lines[i]] = 'instrument'
        i += 1
    while lines[i] != '4':
        if lines[i] == '3' or lines[i] == '':
            i += 1
            continue
        group[lines[i]] = 'culture'
        i += 1
    while lines[i] != '5':
        if lines[i] == '4' or lines[i] == '':
            i += 1
            continue
        group[lines[i]] = 'mood'
        i += 1
    while i < len(lines):
        if lines[i] == '5' or lines[i] == '':
            i += 1
            continue
        group[lines[i]] = 'decade'
        i += 1
    
    with open('dataset.nosync/cluster_extra.txt') as f:
        lines = f.readlines()
    
    extras = dict()
    for line in lines:
        line = line.strip().lower()
        parts = line.split(',')

        if parts[1].strip() == '2':
            extras[parts[0].strip()] = ('instrument', parts[2].strip())
        elif parts[1].strip() == '3':
            extras[parts[0].strip()] = ('culture', parts[2].strip())
        elif parts[1].strip() == '4':
            extras[parts[0].strip()] = ('mood', parts[2].strip())
    
    tags = [group, extras]
    json.dump(tags, open('dataset.nosync/tag_cluster.json', 'w'), indent=4)

def cluster_song():
    with open('dataset.nosync/song_tags.json') as f:
        song_tags = json.load(f)
    
    with open('dataset.nosync/tag_cluster.json') as f:
        cluster, extra = json.load(f)
    
    cluster_tags = dict()
    for song in song_tags:
        cluster_tags[song] = defaultdict(list)
        for tag in song_tags[song]:
            c = cluster[tag]
            if tag not in cluster_tags[song][c]:
                cluster_tags[song][c].append(tag)
            if tag in extra:
                c2, t = extra[tag]
                if t not in cluster_tags[song][c2]:
                    cluster_tags[song][c2].append(t)
    json.dump(cluster_tags, open('dataset.nosync/clustered_song_tags.json', 'w'))

def cluster_tag():
    with open('dataset.nosync/tag_cluster.json') as f:
        tag_cluster, extra = json.load(f)
    clusters = defaultdict(set)
    for t, c in tag_cluster.items():
        clusters[c].add(t)
    for tag in extra:
        c, t = extra[tag]
        clusters[c].add(t)
    for c in clusters:
        clusters[c] = list(clusters[c])
    json.dump(clusters, open('dataset.nosync/clustered_tags.json', 'w'), indent=4)

def stat():
    with open('dataset.nosync/clustered_song_tags.json') as f:
        song_tags = json.load(f)
    
    tag_count = defaultdict(int)
    cluster_count = defaultdict(int)
    for song in song_tags:
        for c in song_tags[song]:
            for tag in song_tags[song][c]:
                tag_count[tag] += 1
            cluster_count[c] += len(song_tags[song][c])
    tag_count = OrderedDict(sorted(tag_count.items(), key=lambda x: x[1], reverse=True))
    cluster_count = OrderedDict(sorted(cluster_count.items(), key=lambda x: x[1], reverse=True))
    json.dump([cluster_count, tag_count], open('dataset.nosync/cluster_stat.json', 'w'), indent=4)


if __name__ == '__main__':
    # cluster()
    # cluster_song()
    # cluster_tag()
    stat()