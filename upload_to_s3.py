import os

from glob import glob
import boto3

import argparse

# Upload downloaded zip to s3
def upload_to_s3(args): 
    zip_list = glob("*.zip")
    print("Zips to upload: ", zip_list)
    for zip_file in zip_list: 
        try:
            s3 = boto3.client("s3")
            s3.upload_file(
                Filename=zip_file, 
                Bucket=args.bucket_name, 
                Key=zip_file
            )
            print("Uploaded resource")

            s3 = boto3.resource("s3")
            s3_file_objs = s3.Bucket(args.bucket_name).objects
            file_uploaded = zip_file in [zip.key for zip in list(s3_file_objs.filter(Prefix=zip_file))]
            print("Checked resource was uploaded")

            # print(file_uploaded)
            if file_uploaded:
                print("Filed uploaded to s3 successfully")
                os.remove(zip_file)
        except Exception as e: 
            print("Error uploading file", e)

if __name__=="__main__":
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument(
        "--completed_file",
        help="Location of CSV with downloaded video ids",
        default="/home/ubuntu/AV-Speech-Dataset/process_data/completed",
        type=str,
    )

    PARSER.add_argument(
        "--data_root",
        help="Root folder of downloaded videos",
        default="/home/ubuntu/AV-Speech-Dataset/process_data/output",
        type=str,
    )

    PARSER.add_argument(
        "--zipped_file",
        help="Location of CSV with zipped video ids",
        default="/home/ubuntu/AV-Speech-Dataset/zip_data/zipped",
        type=str,
    )

    PARSER.add_argument(
        "--bucket_name",
        help="Bucket to upload zipped files to S3",
        default="av-speech-dataset",
        type=str,
    )

    ARGS = PARSER.parse_args()

    upload_to_s3(ARGS)