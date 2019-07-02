import os
import urllib.request
from sh import gunzip
import numpy as np
from sklearn.model_selection import train_test_split
from mlxtend.data import loadlocal_mnist

class FashionMnistLoader:
    
    dir_name = "data/fashion"
    url_train_imgs = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz"
    url_train_labels = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz"
    url_test_imgs = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz"
    url_test_labels = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"
    
    def __init__(self):
        self.train_imgs_fn = None
        self.train_labels_fn = None
        self.test_imgs_fn = None
        self.test_labels_fn = None
        
    def get_data(self, url):
        gz_file_name = url.split("/")[-1]
        gz_file_path = os.path.join(self.dir_name, gz_file_name)  
        file_name = gz_file_name.split(".")[0]
        file_path = os.path.join(self.dir_name, file_name)
        os.makedirs(self.dir_name, exist_ok=True)
        if not os.path.exists(file_path):
            urllib.request.urlretrieve(url, gz_file_path) 
            gunzip(gz_file_path)  
        return file_path
        
    def get_all_data(self):
        self.train_imgs_fn = self.get_data(self.url_train_imgs)
        self.train_labels_fn = self.get_data(self.url_train_labels)
        self.test_imgs_fn = self.get_data(self.url_test_imgs)
        self.test_labels_fn = self.get_data(self.url_test_labels)
        return self
    
    def load_train(self):
        X, y = loadlocal_mnist(
            images_path=self.train_imgs_fn, 
            labels_path=self.train_labels_fn)
        return X, y
    
    def load_test(self):
        X, y = loadlocal_mnist(
            images_path=self.test_imgs_fn, 
            labels_path=self.test_labels_fn)
        return X, y
    
    def _split(self, X, y, test_size):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=666)
        return X_train, X_test, y_train, y_test
            
    def train_split(self, test_size):
        X, y = self.load_train()
        X_train, X_test, y_train, y_test = self._split(X, y, test_size)
        return X_train, X_test, y_train, y_test

    def standard_split(self):
        X_train, y_train = self.load_train()
        X_test, y_test = self.load_test()
        return X_train, X_test, y_train, y_test
    
    
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

transform = transforms.Compose(
    [transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize([0], [1])])

class FashionMnist(Dataset):
    
    def __init__(self, X, y, transform=None):
        self.data = (torch.from_numpy(X).float()/255).reshape(-1, 1, 28, 28).squeeze()
        self.target = torch.from_numpy(y).long()
        self.transform = transform
        
    def __len__(self):
        return len(self.target)
        
    def __getitem__(self, index):
        img, tar = self.data[index], self.target[index]
        if self.transform:
            img = self.transform(img)
        return img, tar