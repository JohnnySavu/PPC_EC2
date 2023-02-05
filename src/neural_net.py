import torch
import torchvision.models as models
import os 
from PIL import Image
import cv2
import torch.nn as nn
import numpy as np 
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import uuid
from datetime import datetime

class Hyperparameter():
    IMAGE_SIZE = (40, 40)
    NUMBER_OF_CLASSES = 43
    BATCH_SIZE = 1

class NeuralNetwork():
    LOCAL_MODEL_PATH = "./data/model.pth"
    BUCKET_MODEL_NAME = "model.pth"
    BUCKET_FOLDER_PATH = "models"

    def __init__(self, s3bucket = None) -> None:
        if not os.path.exists("./data"):
            os.mkdir("./data")
        self.model = models.resnet50(num_classes=Hyperparameter.NUMBER_OF_CLASSES)

        self.data_transformations = {
            "test" : transforms.Compose([transforms.Resize(Hyperparameter.IMAGE_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])]),
            "train" : transforms.Compose([transforms.Resize(Hyperparameter.IMAGE_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])]), 
        }
        self.load_model(s3bucket)
    
    def save_model(self, s3bucket = None, loss_item = None ):
        torch.save(self.model.state_dict(), NeuralNetwork.LOCAL_MODEL_PATH)
        if loss_item is not None:
            loss_id = str(uuid.uuid4())
            with open(f"./data/{loss_id}.txt", "w") as f:
                f.write(f"loss for the train at {datetime.now()} is : {loss_item}")
            
            if s3bucket is not None:
                s3bucket.upload_file(f"./data/{loss_id}.txt", "data", f"{loss_id}.txt")

        if s3bucket is not None:
            s3bucket.upload_file(NeuralNetwork.LOCAL_MODEL_PATH, NeuralNetwork.BUCKET_FOLDER_PATH, NeuralNetwork.BUCKET_MODEL_NAME)

    def load_model(self, s3bucket):
        if s3bucket is not None:
            s3bucket.download_file(os.path.join(NeuralNetwork.BUCKET_FOLDER_PATH, NeuralNetwork.BUCKET_MODEL_NAME), "./data", "model.pth")
            if os.path.exists(NeuralNetwork.LOCAL_MODEL_PATH):
                self.model.load_state_dict(torch.load(NeuralNetwork.LOCAL_MODEL_PATH))

    def predict(self, image):
        #preprocess image 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        self.model.eval()
        image = self.data_transformations['test'](image).unsqueeze(0)
        output = self.model(image)
        return output.argmax().item()
 
    def init_dataloader(self):
        train_data = ImageFolder(root="./train_data", transform=self.data_transformations['train'])
        self.dataloader = DataLoader(train_data, batch_size = Hyperparameter.BATCH_SIZE, shuffle=True, num_workers=1)

    def train(self, s3bucket = None, lr = 1e-3, epochs = 3):
        if s3bucket is not None:
            s3bucket.get_training_data()
        
        self.init_dataloader()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        actual_loss_item = []
        self.model.train()
        for _ in range(epochs):
            actual_loss = []
            for images, targets in self.dataloader:
                images = images
                targets = targets

                self.model.zero_grad()

                with torch.set_grad_enabled(True):
                    output = self.model(images)
                    loss = criterion(output, targets)
                    loss.backward()
                    optimizer.step()
                
                actual_loss.append(loss.item())
        
            actual_loss_item.append(np.array(actual_loss).mean())

        self.save_model(s3bucket, np.array(actual_loss_item).mean())

def unit_tests():
    nn = NeuralNetwork()

if __name__ == "__main__": 
    unit_tests()