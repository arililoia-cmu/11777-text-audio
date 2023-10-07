import urllib.request
import pandas as pd
import csv
from itertools import islice
import sys


def download_csv(start, end):

    # Read csv file
    rows = csv.reader(open('dataset.nosync/ecals_7digital.csv', 'r'))
    rows = islice(rows, start, end)

    records = []
    download = 0
    for i, row in enumerate(rows):
        idx = row[0]
        msdid = row[1]
        release_id = row[3]
        track_id = row[4]
        tag = row[5]
        # Download audio file from 7digital
        try:
            urllib.request.urlretrieve(f"https://us.7digital.com/stream/release/{release_id}/track/{track_id}/m4a", 
                                       f"dataset.nosync/audio/{msdid}.m4a")
            records.append((idx, msdid, tag))
            download += 1
        except:
            # print(f"Error: {msdid}")
            pass
        
        if (i + 1) % 100 == 0:
            print(f'Progress: {download} / {i + 1}')

    # Append to csv file
    with open(f'dataset.nosync/tags.csv', 'a') as f:
        writer = csv.writer(f)
        # writer.writerow(['id', 'msdid', 'tag'])
        writer.writerows(records)


if __name__ == '__main__':
    start, end = int(sys.argv[1]), int(sys.argv[2])
    download_csv(start, end)


