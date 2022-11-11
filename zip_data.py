#!/usr/bin/env python3
import pandas as pd
from zipfile import ZipFile
import os
from os.path import basename, exists
import subprocess 

import argparse

# Number of files to zip
# NUM_TO_ZIP = 10000

# Upload downloaded zip to s3
# def upload_to_s3(args): 
#     zip_list = glob("*.zip")
#     print("Zips to upload: ", zip_list)
#     for zip_file in zip_list: 
#         try:
#             s3 = boto3.client("s3")
#             s3.upload_file(
#                 Filename=zip_file, 
#                 Bucket=args.bucket_name, 
#                 Key=zip_file
#             )
#             print("Uploaded resource")

#             s3 = boto3.resource("s3")
#             s3_file_objs = s3.Bucket(args.bucket_name).objects
#             file_uploaded = zip_file in [zip.key for zip in list(s3_file_objs.filter(Prefix=zip_file))]
#             print("Checked resource was uploaded")

#             # print(file_uploaded)
#             if file_uploaded:
#                 print("Filed uploaded to s3 successfully")
#                 os.remove(zip_file)
#         except Exception as e: 
#             print("Error uploading file", e)
           

def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

# Remove the files that have been zipped already
def remove_zipped_files(args):
    df_to_zip = pd.read_csv(args.completed_file)
    df_zipped = pd.read_csv(args.zipped_file)
    # Delete from previous endpoint to current endpoint
    start_idx = df_zipped.iloc[-2].total_zipped
    for index, row in df_to_zip[start_idx:start_idx+args.num_to_zip].iterrows(): 
        file_path = args.data_root + "/{}-{}-{}.m4a".format(row['id'], row['start'], row['end'])
        os.remove(file_path)
        print("%s has been removed successfully" %file_path)

def main(args):
    df_to_zip = pd.read_csv(args.completed_file)
    df_zipped = pd.read_csv(args.zipped_file)

    total_zipped = 0
    if len(df_zipped) > 0: 
        zipped_last = df_zipped.iloc[-1]
        total_zipped = zipped_last['total_zipped']

    # Starting file to ending file in zip (inclusive)
    id_start = df_to_zip.iloc[total_zipped]['id'][-16:]
    id_end = df_to_zip.iloc[total_zipped + args.num_to_zip-1]['id'][-16:]

    try: 
        zip_file_name = "{}-{}-{}.zip".format(id_start, id_end, args.num_to_zip)
        zip_obj = ZipFile(zip_file_name, 'w')

        for index, row in df_to_zip[total_zipped:total_zipped+args.num_to_zip].iterrows(): 
            file_path = args.data_root + "/{}-{}-{}.m4a".format(row['id'], row['start'], row['end'])
            # if not exists(file_path):
            #     print("PATH NO EXIST: ", file_path)
            #     continue
            actual_len = get_length(file_path)
            pred_len = float(row['end']) - float(row['start'])
            diff = abs(actual_len - pred_len)

            if diff > 2: 
                print("ERROR", row['id'], diff, pred_len, actual_len)
            else: 
                zip_obj.write(file_path, basename(file_path))
                print("Zipped", index - total_zipped, "Files")
                print("Progress (%): ", (index - total_zipped) / args.num_to_zip)
        
        zip_obj.close()

        try: 
            zipped_log = open(args.zipped_file, 'a')
            log = "{},{},{},{},{}\n".format(
                id_start, 
                id_end, 
                args.num_to_zip,
                total_zipped + args.num_to_zip,
                zip_file_name
            )
            zipped_log.write(log)
        except Exception as e: 
            print("Error writing to log", e)
        
    except Exception as e: 
        print("Unsuccessful zipping of files", e)


if __name__=="__main__":
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument(
        "--completed_file",
        help="Location of CSV with downloaded video ids",
        default="./completed",
        type=str,
    )

    PARSER.add_argument(
        "--data_root",
        help="Root folder of downloaded videos",
        default="./output",
        type=str,
    )

    PARSER.add_argument(
        "--zipped_file",
        help="Location of CSV with zipped video ids",
        default="./zipped",
        type=str,
    )

    PARSER.add_argument(
        "--bucket_name",
        help="Bucket to upload zipped files to S3",
        default="cs543-spotify-data",
        type=str,
    )

    PARSER.add_argument(
        "--num_to_zip",
        help="Number of files to Zip",
        default=10000,
        type=int,
    )

    ARGS = PARSER.parse_args()
    main(ARGS)

    # Remove m4a files after they have been zipped successfully
    remove_zipped_files(ARGS)

    # Upload downloaded zip to s3
    # upload_to_s3(ARGS)
