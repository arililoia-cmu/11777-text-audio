import urllib.request
import csv
from itertools import islice
import sys
import time
import requests
import concurrent.futures


def download_audio(row):
    idx = row[0]
    msdid = row[1]
    release_id = row[3]
    track_id = row[4]
    tag = row[5]
    
    # Download audio file from 7digital
    try:
        url = f"https://us.7digital.com/stream/release/{release_id}/track/{track_id}/m4a"
        output = f"audio/{msdid}.m4a"
        
        # Using urlretrieve
        urllib.request.urlretrieve(url, output)
        return (idx, msdid, tag), 1

        # Using requests.get
        # response = s.get(url)
        # if response.status_code == 200:
        #     with open(output, 'wb') as f:
        #         f.write(response.content)
        #     return (idx, msdid, tag), 1
        # else:
        #     return (idx, msdid, tag), 0
    except:
        # print(f"Error: {msdid}")
        return (idx, msdid, tag), 0


def download_csv(start, end):

    # Read csv file
    rows = csv.reader(open('ecals_7digital.csv', 'r'))
    rows = islice(rows, start, end)

    records = []
    download = 0
    # s = requests.Session()
    total = 0

    # Use ThreadPoolExecutor to download concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_audio, row) for row in rows]
        for future in concurrent.futures.as_completed(futures):
            data, success = future.result()
            if success:
                records.append(data)
                download += 1
            total += 1
            if (total + 1) % 100 == 0:
                print(f'Progress: {download} / {total + 1}')

    # Append to csv file
    records.sort(key=lambda x: x[0])
    with open(f'song_tags.csv', 'a') as f:
        writer = csv.writer(f)
        # writer.writerow(['id', 'msdid', 'tag'])
        writer.writerows(records)


if __name__ == '__main__':
    start, end = int(sys.argv[1]), int(sys.argv[2])
    begin = time.time()
    download_csv(start, end)
    print(f'Time elapsed: {(time.time() - begin) / 60:.2f} minutes')
