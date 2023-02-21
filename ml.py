from flask import Flask, render_template
# from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import cv2
import torch
from torch import nn
import torch.nn.functional as F
import skorch
from skorch import NeuralNetClassifier
from IPython.core.display import HTML

app = Flask(__name__)

@app.route("/prof")
def prof():
    return render_template('profview.html')

@app.route("/data")
def scan():
    mnist = fetch_openml('mnist_784', as_frame=False, cache=False)

    X = mnist.data.astype('float32')
    y = mnist.target.astype('int64')

    X /= 255
    X.shape

    XCnn = X.reshape(-1, 1, 28, 28)
    XCnn_train, XCnn_test, y_train, y_test = train_test_split(XCnn, y, test_size=0.25, random_state=42)

    mnist_dim = X.shape[1]
    hidden_dim = int(mnist_dim/8)
    output_dim = len(np.unique(mnist.target))
    mnist_dim, hidden_dim, output_dim

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    class ClassifierModule(nn.Module):
        def __init__(
                self,
                input_dim=mnist_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=0.5,
        ):
            super(ClassifierModule, self).__init__()
            self.dropout = nn.Dropout(dropout)

            self.hidden = nn.Linear(input_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, output_dim)

        def forward(self, X, **kwargs):
            X = F.relu(self.hidden(X))
            X = self.dropout(X)
            X = F.softmax(self.output(X), dim=-1)
            return X

    class Cnn(nn.Module):
        def __init__(self, dropout=0.5):
            super(Cnn, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
            self.conv2_drop = nn.Dropout2d(p=dropout)
            self.fc1 = nn.Linear(1600, 100) # 1600 = number channels * width * height
            self.fc2 = nn.Linear(100, 10)
            self.fc1_drop = nn.Dropout(p=dropout)

        def forward(self, x):
            x = torch.relu(F.max_pool2d(self.conv1(x), 2))
            x = torch.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            
            # flatten over channel, height and width = 1600
            x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
            
            x = torch.relu(self.fc1_drop(self.fc1(x)))
            x = torch.softmax(self.fc2(x), dim=-1)
            return x

    torch.manual_seed(0)

    cnn = NeuralNetClassifier(
        Cnn,
        max_epochs=10,
        lr=0.002,
        optimizer=torch.optim.Adam,
        device=device,
    )
    cnn.fit(XCnn_train, y_train);

    filename = "mod.sav"
    pickle.dump(cnn,open(filename, 'wb'))

    img = cv2.imread('download.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('downloadk.png', cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(255-img, (28, 28))
    img2 = cv2.resize(255-img2, (28, 28))

    (thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    (thresh2, img2) = cv2.threshold(img2, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    plt.imshow(img2, cmap="gray")

    print(img.shape)
    X = img.reshape(1, 784)
    X = X.astype('float32')
    X /= 255.0
    XCnn = X.reshape(-1, 1, 28, 28)

    X2 = img2.reshape(1, 784)
    X2 = X2.astype('float32')
    X2 /= 255.0
    XCnn2 = X2.reshape(-1, 1, 28, 28)

    loaded_model = pickle.load(open('/content/mod (1).sav', 'rb'))
    result = loaded_model.predict(XCnn)

    result2 = loaded_model.predict(XCnn2)

    results = [result, result2]
    
    return results
