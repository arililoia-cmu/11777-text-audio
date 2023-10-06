import h5py
import urllib.request
import sqlite3
import pandas as pd
import csv
from itertools import islice
import sys


def download_h5():
    h5 = h5py.File('msd_summary_file.h5', 'r')
    metadata = h5['metadata']['songs']
    success = 0
    for i in range(1000, 2000):
        song = metadata[i]
        release_id = song[15]
        track_id = song[19]
        song_id = str(song[17])[2:-1]
        if release_id != '' and track_id != '':
            try:
                urllib.request.urlretrieve(f"https://us.7digital.com/stream/release/{release_id}/track/{track_id}/m4a", f"Downloads/msd/{song_id}.m4a")
                success += 1
            except:
                # print(f"Error: {song_id}")
                pass

def download_sql(start, end):
    # Create a connection to the SQL database
    conn = sqlite3.connect('track_metadata.db')

    # Create a cursor
    cursor = conn.cursor()

    # Read ecals_7digital view
    cursor.execute('SELECT * FROM ecals_7digital WHERE id BETWEEN ? AND ?', (start, end))
    rows = cursor.fetchall()
    print('Rows fetched:', len(rows))

    records = []
    success = 0
    for i, row in enumerate(rows):
        idx = row[0]
        msdid = row[1]
        release_id = row[3]
        track_id = row[4]
        try:
            urllib.request.urlretrieve(f"https://us.7digital.com/stream/release/{release_id}/track/{track_id}/m4a", f"audio/{msdid}.m4a")
            records.append((idx, msdid, release_id, track_id))
            success += 1
        except:
            # print(f"Error: {msdid}")
            pass
        
        if (i + 1) % 100 == 0:
            print(f'Progress: {success} / {i + 1}')
    
    df = pd.DataFrame(records, columns=['id', 'msdid', 'release_id', 'track_id'])
    df.to_json(f'downloads_{start}_{end}.json')

def download_csv(start, end):
    # Read csv file
    rows = csv.reader(open('ecals_7digital.csv', 'r'))
    rows = islice(rows, start, end)

    records = []
    success = 0
    for i, row in enumerate(rows):
        idx = row[0]
        msdid = row[1]
        release_id = row[3]
        track_id = row[4]
        try:
            urllib.request.urlretrieve(f"https://us.7digital.com/stream/release/{release_id}/track/{track_id}/m4a", f"audio/{msdid}.m4a")
            records.append((idx, msdid, release_id, track_id))
            success += 1
        except:
            # print(f"Error: {msdid}")
            pass
        
        if (i + 1) % 100 == 0:
            print(f'Progress: {success} / {i + 1}')

    with open(f'downloads_{start}_{end}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'msdid', 'release_id', 'track_id'])
        writer.writerows(records)


if __name__ == '__main__':
    start, end = int(sys.argv[1]), int(sys.argv[2])
    download_csv(start, end)
