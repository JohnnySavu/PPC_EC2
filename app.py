from flask import Flask, request, Response
import src.constants as constants
import jsonpickle
import numpy as np
import cv2
import logging
from src.aws_bucket import S3Bucket
from src.neural_net import NeuralNetwork
import uuid
import os 

app = Flask(__name__)
bucket_storage = S3Bucket()
neural_network = NeuralNetwork()
    

def configure_logging():
    logging.basicConfig(filename='log_file.txt', encoding='utf-8', level=logging.DEBUG)

def create_response(message : str, status_code : int):
    response_msg = jsonpickle.encode({"message" : message})
    return Response(response_msg, status=status_code, mimetype="application/json")

'''
receives an image and returns the corresponding label, from 0 to 42 
'''
@app.route("/predict", methods = ["POST"])
def predict():
    try:
        img_nparray = np.fromstring(request.data, np.uint8)
        image = cv2.imdecode(img_nparray, cv2.IMREAD_COLOR)
        prediction = neural_network.predict(image)
        
    except Exception as exception:
        logging.exception(str(exception))
        return Response("Error", status=500)

    response_msg = f'{prediction}'
    return create_response(response_msg, 200)


'''
trains the latest model and saves it in s3 bucket 
'''
@app.route("/train", methods = ['PUT'])
def train():
    nn = NeuralNetwork()
    nn.train(bucket_storage)
    return create_response("ok", 200)

'''
adds a new image for training the model 
'''
@app.route("/add_data", methods = ['PUT'])
def add_image():
    try:
        correct_label = request.args.get("label")
        img_nparray = np.fromstring(request.data, np.uint8)
        image = cv2.imdecode(img_nparray, cv2.IMREAD_COLOR)

        img_name = str(uuid.uuid4())
        cv2.imwrite(f"./{img_name}.jpg", image)
        bucket_storage.upload_file(f"./{img_name}.jpg", correct_label, f"{img_name}.jpg")
        os.remove(f"./{img_name}.jpg")

        response_msg = f'ok'
        return create_response(response_msg, 200)
    except Exception as exception:
        logging.exception(str(exception))
        return Response("Error", status=500)

'''
gets data needed for the quiz 
'''
@app.route("/quizz_data", methods = ['GET'])
def get_quizz_data():
    try:
        response = bucket_storage.get_quiz_data()
        
        return Response(jsonpickle.encode(response),  status=200, mimetype="application/json")
    except Exception as e:
        logging.error(e)
        return Response(jsonpickle.encode({}), status=500)

@app.route("/", methods=['GET'])
def welcome():
    return "Hello World"

if __name__ == '__main__':
    configure_logging()
    app.run(host = constants.HOST, port = constants.PORT, debug=True)
