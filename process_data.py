#!/usr/bin/env python3
# Data Processing File
# Goal: Process cleaned videos from spotify_train
# Run with proxychains: proxychains python3 scraper.py
# Logs:
# 1. logs folder - stores debug results from each run
# 2. completed file - stores ids of successful runs (csv format)
# 3. downloaderrors file - stores ids of failed runs (csv format)
# 4. output folder - stores songs

# from tqdm.contrib.concurrent import process_map
# import tqdm
import subprocess
import os
from multiprocessing import Pool, freeze_support
import numpy as np
import pandas as pd
import time
import random
import http.client
from datetime import datetime as dt

MAX_RETRIES = 2
NUM_WORKERS = 4
RERUN_FAILED = False
RERUN_INCOMPLETE_READS = False

# How many iterations we have done 
# through the entire dataset
RUN_NUM = 2

def filter_completed_videos(df_train, ids_file): 
    # Read Video ids
    df_completed = pd.read_csv(ids_file)[['id', 'start', 'end']]
    # Remove Videos from Training Videos
    return pd.concat([df_train, df_completed, df_completed]).drop_duplicates(subset=['id', 'start', 'end'], keep=False)

def filter_unplayable_videos(df_train, ids_file): 
    # Read Video ids
    df_unplayable = pd.read_csv(ids_file)
    # Remove Videos from Training Videos
    return df_train[~df_train['id'].isin(df_unplayable['id'])]

def filter_found_unplayable_videos(df_train, ids_file): 
    # Read Video Ids
    df_unplayable = pd.read_csv(ids_file, on_bad_lines = 'warn')
    df_unplayable = df_unplayable[(df_unplayable['error reason'] == 'Private') | \
                (df_unplayable['error reason'] == 'Unavailable') | \
                (df_unplayable['error reason'] == 'Member-Only Access') | \
                (df_unplayable['error reason'] == 'Age-restricted') | \
                (df_unplayable['error reason'] == 'Youtube ToS Violation') | \
                (df_unplayable['error reason'] == 'Invalid data found - no moov atom') | \
                (df_unplayable['error reason'] == 'Requested format unavailable') | \
                (df_unplayable['error reason'] == 'members-only content') | \
                (df_unplayable['error reason'] == 'Conversion Failed') ] # | \
                # Included temporarily
                # (df_unplayable['error reason'] == 'HTTP Incomplete Read') ]# | \
                # (df_unplayable['error reason'] == 'Unknown')]

    return df_train[~df_train['id'].isin(df_unplayable['id'])]

def write_completed(song_name, ytid, start, end, num_retries, start_time):
    completed = open('completed', 'a')
    total_time = time.time() - start_time
    log = f"{song_name},{ytid},{start},{end},{num_retries},{dt.now()},{total_time},{RUN_NUM}\n"
    completed.write(log)

def write_download_error(log_error, song_name, spurl, start, end, num_retries):
    print("Download Error", spurl)
    reason = 'Unknown'
    if 'Server returned 403 Forbidden (access denied)' in log_error: 
        reason = 'Cache Issue (re-run)'
    elif 'ERROR: Video unavailable' in log_error or 'ERROR: This video is not available' in log_error:
        reason = 'Unavailable'
    elif 'ERROR: Private video' in log_error:
        reason = 'Private'
    elif 'ERROR: Join this channel to get access' in log_error: 
        reason = 'Member-Only Access'
    elif 'ERROR: Sign in to confirm your age' in log_error: 
        reason = 'Age-restricted'
    elif 'ERROR: This video has been removed' in log_error: 
        reason = 'Youtube ToS Violation'
    elif 'ERROR: giving up after 10 retries' in log_error:
        reason = 'Proxy Failed'
    elif 'moov atom not found' in log_error: 
        reason = 'Invalid data found - no moov atom'
    elif 'Invalid data found' in log_error: 
        reason = 'Invalid data found'
    elif 'http.client.IncompleteRead' in log_error: 
        reason = 'HTTP Incomplete Read'
    elif 'ERROR: requested format not available' in log_error: 
        reason = 'Requested format unavailable'
    elif 'members-only content' in log_error:
        reason = 'members-only content'
    elif 'ERROR: Conversion failed!' in log_error: 
        reason = 'Conversion Failed'
    elif 'codec not currently supported' in log_error: 
        reason = 'Codec not supported'

    download_error = open('downloaderrors', 'a')
    log = song_name + ',' + spurl + ',' + str(start) + ',' + str(end) + ',' + reason + ',' + str(num_retries) + '\n'
    download_error.write(log)

def rerunnable(log_error): 
    if "http.client.IncompleteRead" in log_error: 
        return True
    elif "Server returned 403 Forbidden (access denied)" in log_error:
        return True
    return False

def process(t):
    if not RERUN_FAILED and not RERUN_INCOMPLETE_READS:
        spid, song_name, spurl ,start, end = t
    else: 
        spurl, start, end, error_reason = t
    
    # start, end = 0, 30
    start_time = time.time()
    # Skipped
    if os.path.isfile('/home/ubuntu/AV-Speech-Dataset/process_data/output/{}-{}-{}.mp3'.format(spid, start, end)):
        print("Video Skipped")
        write_completed(song_name, spurl, start, end, np.nan, start_time)
        return

    cmd = [
        # 'proxychains',
        'spotdl',
        f'{spurl}',
        '--cookie-file', 'cookies.txt',
        '--ffmpeg', 'ffmpeg', 
        '--ffmpeg-args', "-y -loglevel error -ss {} -to {} -strict experimental -f mp3 -threads 1".format(start, end),
        '--bitrate', "128k",
        '--format', 'mp3' ,
        '--output', "output/{}-{}-{}.mp3".format(spid, start, end),
    ]

    print('CMD', cmd)

    # Log this command use fileio
    log_file = open('logs/{}-{}-{}'.format(spid, start, end), 'w+')
    log_file.write(str(cmd))
    log_result = ''
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            # Random delay to avoid incomplete reads from proxy overloading - not working
            # time.sleep(0.5 * random.randint(1, 5))
            log_result = subprocess.run(cmd, capture_output=True, text=True)
            print(log_result)
            # For Command Line Logging
            # result = subprocess.run(cmd, capture_output=True, text=True)
            print("STDOUT: ", log_result.stdout, spurl, '\n')
            print("STDERR: ", log_result.stderr, spurl, '\n')

            log_file.write(log_result.stdout)
            log_file.write(log_result.stderr)

            output_folder = "./output"
            output_file = "/{}-{}-{}.mp3".format(spid, start, end)
            if os.path.exists(output_folder + output_file):
                write_completed(song_name, spurl, start, end, attempt, start_time)
            else:
                attempt += 1
                if attempt < MAX_RETRIES and rerunnable(log_result.stderr):
                    print("Retrying", spurl)
                    continue
                print(f"All {MAX_RETRIES} retries failed, {spurl}")
                write_download_error(log_result.stderr, song_name, spurl, start, end, attempt)

        except Exception as exception:
            print("Process Exception Raised", spurl)
            print("Type: ", type(exception).__name__)
            attempt += 1
            if attempt == MAX_RETRIES:
                print(f"All {MAX_RETRIES} retries failed, {spurl}")
                log_file.write("All retries failed")
                log_file.write("Process Failed")
                log_file.write(log_result.stdout)
                log_file.write(log_result.stderr)
                write_download_error(log_result.stderr, song_name, spurl, start, end, attempt)
        else:
            break

def main():
    # https://www.codetd.com/en/article/10831740 ?
    # http.client.HTTPConnection._http_vsn = 10
    # http.client.HTTPConnection._http_vsn_str = 'HTTP/1.0'

    df_train = None
   
    # Read from training data
    df_train = pd.read_csv('spotify.csv')
    # Original size
    print('Length of original training set: ', len(df_train))
    # Filter videos unplayable found from past download errors
    df_train = filter_found_unplayable_videos(df_train, './downloaderrors')
    print('Length found removed unplayables: ', len(df_train))
    # Filter Completed Videos 
    df_train = filter_completed_videos(df_train, './completed')
    print('Length removed completed: ', len(df_train))

    # Get the start time
    start_time = time.time()
    # df_train['start_time'] = start_time

    # Convert this dataframe into a list of rows
    input_list = df_train.to_numpy().tolist()
    total_tasks = len(input_list)

    # process(input_list[2])

    # With TDQM new
    # r = process_map(process, input_list, max_workers=2)

    # With TQDM Old
    #with Pool(15) as p:
     # r = list(tqdm.tqdm(p.imap_unordered(process, input_list), total=len(input_list)))

    # Without TQDM
    p = Pool(NUM_WORKERS)
    # Unordered
    
    for i, _ in enumerate(p.map(process, input_list), 1):
        elapsed_time = time.time() - start_time
        print("Audio clips Done: ", i)
        print("Elapsed Time", elapsed_time)
        print("Avg audio clips per second", i / elapsed_time)
        print("Progress made: ", i / total_tasks)

        # if i % 400 == 0:
        #     cmd = 'find ./output -name "*.part" -or -name "*.ytdl" -or -name "*.f140" | xargs rm'
        #     print("Removing failed files")
        #     subprocess.Popen(cmd, shell=True)
        if i % 12000 == 0:
            cmd = 'python3 ./zip_data.py'
            print("zipping data")
            #zip_result = subprocess.run(cmd.split(' '), capture_output=True, text=True)
            #print(zip_result)
            subprocess.Popen(cmd.split(' '))
    
    #cmd = 'python3 ../zip_data/zip_data.py'
    #subprocess.Popen(cmd.split(' '))
    # res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    #print(res)
    # Ordered
    # p.map(process, input_list)
    
    p.close()
    p.join()
    print("Pool time: ", time.time() - start_time)

if __name__=="__main__":
    # freeze_support() # optional if the program is not frozen
    main()
