import mlflow
import mlflow.pytorch
import os
import boto3
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base
from datasets import Dataset
import ModelTraining
import Preprocessing
from queue import Queue
import json
import threading
from PyPDF2 import PdfReader
from io import BytesIO

load_dotenv("./env/.env")

DATABASE_URL = os.getenv('DATABASE_URL')

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI) 
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = automap_base()
Base.prepare(engine=engine, reflect=True, schema="public")
Used_files = Base.classes.used_files

def get_all_s3_keys():
    s3_client = boto3.client('s3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME)

    contents_keys = []

    for contents in response['Contents']:
        contents_keys.append(contents['Key'])
    
    return contents_keys

def get_all_used_files():
    db = SessionLocal()
    used_filenames = db.query(Used_files.filename).all()
    used_filenames = [filename[0] for filename in used_filenames]
    db.close()

    return used_filenames

def background_finetune(text):
        json_list = Preprocessing.text_to_jsonl_dataset(text)
        dataset_list = [json.loads(line) for line in json_list]
        dataset = Dataset.from_list(dataset_list)
        model, tokenizer = ModelTraining.load_distilgpt2_model()
        tokenizer.pad_token = tokenizer.eos_token
        tokenized_dataset = Preprocessing.prepare_dataset(dataset, tokenizer)
        ModelTraining.finetune(model, tokenizer, tokenized_dataset)

train_queue = Queue()

def worker():
    while True:
        text = train_queue.get()
        if text is None:
            break
        try:
            background_finetune(text)
        except Exception as e:
            print("Eğitim hatası:", e)
        finally:
            train_queue.task_done()

threading.Thread(target=worker, daemon=True).start()

def start_training():
    db = SessionLocal()
    s3_client = boto3.client('s3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    contents_keys = get_all_s3_keys()
    used_filenames = get_all_used_files()

    not_trained = [filename for filename in contents_keys if filename not in used_filenames]

    if(not_trained):
        print("Bütün Dosyalar ile fine tune edilmiş")
    
    for object_name in not_trained:
        used_file = Used_files(filename=object_name)
        db.add(used_file)
        db.commit()

        pdf = s3_client.get_object(Bucket=BUCKET_NAME, Key=object_name)['Body']
        reader = PdfReader(BytesIO(pdf.read()))
        text = Preprocessing.pdf_to_text(reader)

        train_queue.put(text)

    db.close()

if __name__ == "__main__":
    start_training()