from base64 import encodebytes
import io
import logging
import boto3
import os
import shutil
from dotenv import load_dotenv, dotenv_values
import random
from PIL import Image

class Utils():
    def get_response_image(image_path):
        pil_img = Image.open(image_path, mode='r') # reads the PIL image
        byte_arr = io.BytesIO()
        pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
        encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
        return encoded_img

    def get_label(path):
        label = int(path.split("/")[-2])
        return label


class S3Bucket():
    BUCKET_NAME = 'ppcbucket'

    def __init__(self) -> None:
        load_dotenv()
        secrets = dotenv_values(".env")
        self.resource = boto3.resource(
                        service_name='s3',
                        region_name='eu-central-1',
                        aws_access_key_id=secrets["aws_access_key_id"],
                        aws_secret_access_key=secrets["aws_secret_access_key"])
                        
        self.client = self.resource.meta.client
        

    def upload_file(self, file_path : str, bucket_folder :str, bucket_file_name : str):
        key_path = os.path.join(bucket_folder, bucket_file_name)
        try:
            self.client.upload_file(file_path, S3Bucket.BUCKET_NAME, key_path)
        except Exception as e:
            logging.error(e)

    def download_file(self, bucket_path : str, local_dir : str, local_file_name : str):
        try:
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            path = os.path.join(local_dir, local_file_name)
            with open(path, 'wb') as data:
                self.client.download_fileobj(S3Bucket.BUCKET_NAME, bucket_path, data)
        except Exception as e:
            os.remove(path)

    def remove_file(self, bucket_path : str):
        try:
            self.client.delete_object(Bucket=S3Bucket.BUCKET_NAME, Key=bucket_path)
        except Exception as e:
            logging.error(e)

    def move_file(self, initial_path : str, output_path : str):
        try:
            copy_source = {
                'Bucket': S3Bucket.BUCKET_NAME,
                'Key': initial_path
            }
            self.client.copy(copy_source, S3Bucket.BUCKET_NAME, output_path)
            self.remove_file(initial_path)
        except Exception as e:
            logging.log(e)

    def get_training_data(self):
        train_data = []
        for item in self.resource.Bucket(S3Bucket.BUCKET_NAME).objects.all():
            item_path = item.key
            if "train_data" in item_path and ".jpg" in item_path:
                train_data.append(item_path)

        if os.path.exists("./train_data"):
            shutil.rmtree("./train_data")

        for item_path in train_data: 
            path = item_path.split("/")[:-1]
            file_name = item_path.split("/")[-1]
            path = "/".join(path)
            self.download_file(f"{path}/{file_name}", path, file_name)            

    def get_quiz_data(self):
        labeled_candidate_data = []
        unlabeled_candidate_data = []
        for item in self.resource.Bucket(S3Bucket.BUCKET_NAME).objects.all():
            item_path = item.key
            if ".jpg" in item_path and "quiz_labeled_data" in item_path:
                labeled_candidate_data.append(item_path)
            elif ".jpg" in item_path and "quiz_unlabeled_data" in item_path:
                unlabeled_candidate_data.append(item_path)

        unlabeled_data = [random.choice(unlabeled_candidate_data)]
        labeled_data = [random.choice(labeled_candidate_data)]
        labeled_candidate_data = [item for item in labeled_candidate_data if item not in labeled_data]
        labeled_data.append(random.choice(labeled_candidate_data)) 

        for cnt, item in enumerate(unlabeled_data):
            self.download_file(item, "./temp", f"unlabeled_{cnt}.jpg")
        for cnt, item in enumerate(labeled_data):
            self.download_file(item, "./temp", f"labeled_{cnt}.jpg")
        
        response = dict()
        for cnt, item in enumerate(unlabeled_data):
            response[f"image_{cnt}"] = Utils.get_response_image(f"./temp/unlabeled_{cnt}.jpg")
            response[f"label_image_{cnt}"] = "None" 

        for cnt, item in enumerate(labeled_data):
            response[f"image_{cnt}"] = Utils.get_response_image(f"./temp/labeled_{cnt}.jpg")
            response[f"label_image_{cnt}"] = Utils.get_label(item) 

        return response 

def unit_tests():
    obj = S3Bucket()
    obj.upload_file("./../requirments.txt", "old_training_data", "requirements.txt")
    obj.remove_file("old_training_data/requirements.txt")
    obj.download_file("test/req.txt", "ceva.txt")
    obj.remove_file("test/req.txt")