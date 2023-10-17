import pandas as pd
import sqlite3


def h5_sql():
    # Read the .h5 file into a pandas dataframe
    df = pd.read_hdf('msd_summary_file.h5', key='metadata/songs')

    # # Create a connection to the SQL database
    conn = sqlite3.connect('track_metadata.db')
    print(conn)

    # # Write the dataframe to the SQL database
    df.to_sql('summary', conn, if_exists='replace', index=False)

    # # Close the connection to the SQL database
    conn.close()

def json_sql():
    # Read the .json file into a pandas dataframe
    df = pd.read_json('ECALS/ecals_annotation/annotation.json', orient='index')
    
    # Convert tag, release, artist_name, title, track_id to string
    df = df.astype({'tag': 'string', 'release': 'string', 'artist_name': 'string', 'title': 'string', 'track_id': 'string'})
    print(df.dtypes)


    # Create a connection to the SQL database
    conn = sqlite3.connect('track_metadata.db')
    print(conn)

    # Write the dataframe to the SQL database
    df.to_sql('ecals', conn, if_exists='replace', index=False)

    # Close the connection to the SQL database
    conn.close()

def json_csv():
    # Read the .json file into a pandas dataframe
    df = pd.read_json('downloads_1_1000.json', orient='index')

    # Transpose the dataframe
    df = df.transpose()
    
    # Write the dataframe to a .csv file
    df.to_csv('downloads_0_1000.csv', index=False)

def combine_csv():
    # Read full csv into a pandas dataframe
    full = pd.read_csv('ecals_7digital.csv')

    # Read partial csv into a pandas dataframe
    partial = pd.read_csv('downloads_2000_3000.csv')

    # Find the entry that exists in both dataframes based on full['track_id'] and partial['msdid']
    entries = full[full['track_id'].isin(partial['msdid'])]

    # Extract 'id', 'track_id', 'tag'
    entries = entries[['id', 'track_id', 'tag']]

    # Output to a .csv file
    entries.to_csv('entries.csv', index=False, mode='a')

    print(entries)

combine_csv()
