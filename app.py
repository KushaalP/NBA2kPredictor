from flask import Flask, render_template, request
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the dataset
Rating = pd.read_csv("dataset/nba_rankings_2014-2020.csv")

# Select important stats
ImportantStats = ['AGE', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK']

xData = Rating[ImportantStats]
yData = Rating[['rankings']]

# Convert to tensors
xData = torch.tensor(xData.values, dtype=torch.float)
yData = torch.tensor(yData.values, dtype=torch.float)

# Normalize data
meanx = xData.mean(axis=0)
stdx = xData.std(axis=0)
xData = (xData - meanx) / stdx

meany = yData.mean(axis=0)
stdy = yData.std(axis=0)
yData = (yData - meany) / stdy

# Neural Network Definition
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(19, 15)
        self.layer2 = nn.Linear(15, 10)
        self.layer3 = nn.Linear(10, 1)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters())
        self.metrics = {'loss': [], 'acc': []}

    def forward(self, x):
        hidden1 = torch.sigmoid(self.layer1(x))
        hidden2 = torch.sigmoid(self.layer2(hidden1))
        output = self.layer3(hidden2)
        return output

    def train(self, x, y, epochs, batch_size=64):
        for epoch in range(epochs):
            current_loss, n_steps = 0, 0
            batch_idxs = np.random.permutation(range(len(y)))
            for batch_start in range(0, len(y), batch_size):
                batch_idx = batch_idxs[batch_start:min(batch_start + batch_size, len(y))]
                batch_x, batch_y = x[batch_idx], y[batch_idx]
                self.optimizer.zero_grad()
                prediction = self(batch_x)
                loss = self.loss_fn(prediction, batch_y)
                loss.backward()
                current_loss += float(loss.detach().numpy())
                self.optimizer.step()
                n_steps += 1
            self.metrics['loss'].append(current_loss / n_steps)
        return self.metrics

# Train the network
Network = NeuralNetwork()
Network.train(xData, yData, epochs=225)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Extract stats from form
        stats = [float(request.form[stat]) for stat in ImportantStats]
        stats_tensor = torch.tensor([stats])
        
        # Normalize the stats
        stats_normalized = (stats_tensor - meanx) / stdx
        
        # Predict using the model
        prediction = Network(stats_normalized)
        predicted_rating = (prediction * stdy + meany).item()
        
        return render_template('index.html', predicted_rating=round(predicted_rating))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
