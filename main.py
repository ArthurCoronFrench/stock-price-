import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import yfinance as yf

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ticker = 'AAPL'     # ticker symbol for Apple
ticker = 'MSFT'     # Microsoft
ticker = 'GOOGL'    # Google
ticker = 'AMZN'     # Amazon
ticker = 'TSLA'     # Tesla

df = yf.download(ticker, '2020-01-01')      # data since january 2020
print(df.head())

df.Close.plot(figsize=(10,6))

scaler = StandardScaler()
df['Close'] = scaler.fit_transform(df['Close'])     # normalisation: (value - mean) / deviation
df.Close.plot(figsize=(10,6))

seq_length = 30     # 30 days sequence
data = []

for i in range(len(df) - seq_length):       # sliding window
    data.append(df.Close[i:i+seq_length])

data = np.array(data)

train_size = int(0.8*len(data))

X_train = torch.from_numpy(data[:train_size, :-1,:]).type(torch.Tensor).to(device)
y_train = torch.from_numpy(data[:train_size, -1,:]).type(torch.Tensor).to(device)
X_test = torch.from_numpy(data[train_size:, :-1,:]).type(torch.Tensor).to(device)
y_test = torch.from_numpy(data[train_size:, -1,:]).type(torch.Tensor).to(device)

class PredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PredictionModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        
        out, (hn,cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        
        return out
    
model = PredictionModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("#############################")
print("\n")
print("loss: ")
print("\n")
print("#############################")
epochs = 200
for i in range(epochs):
    y_train_pred = model(X_train)
    loss = criterion(y_train_pred, y_train)
    if i % 10 == True:
        print(i, loss.item())
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
model.eval()    # put the model on eval mode
y_test_pred = model(X_test)

print("#############################")
print("\n")
print("Type change: ")
print("\n")
print("#############################")

print(type(y_train_pred))
print(type(y_train))
print(type(y_test_pred))
print(type(y_test))

y_train_pred = scaler.inverse_transform(y_train_pred.detach().cpu().numpy())
y_train = scaler.inverse_transform(y_train.detach().cpu().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().cpu().numpy())
y_test = scaler.inverse_transform(y_test.detach().cpu().numpy())

print(type(y_train_pred))
print(type(y_train))
print(type(y_test_pred))
print(type(y_test))

train_rmse = root_mean_squared_error(y_train[:,0], y_train_pred[:,0])
test_rmse = root_mean_squared_error(y_test[:,0], y_test_pred[:,0])

print("#############################")
print("\n")
print("Train and test RMSE: ")
print("\n")
print("#############################")

print("train rmse: ", train_rmse)
print("test_rmse:", test_rmse)

fig = plt.figure(figsize=(12,10))
gs = fig.add_gridspec(4, 1)
ax1 = fig.add_subplot(gs[:3,0])
ax1.plot(df.iloc[-len(y_test):].index, y_test, color = 'blue', label = 'Actual price')
ax1.plot(df.iloc[-len(y_test):].index, y_test_pred, color = 'green', label = 'Predicted price')
ax1.legend()
plt.title(f"{ticker} Stock price prediction")
plt.xlabel('Date')
plt.ylabel('Price')

ax2 = fig.add_subplot(gs[3, 0])
ax2.axhline(test_rmse, color = 'blue', linestyle = '--', label = 'RMSE')
ax2.plot(df[-len(y_test):].index, abs(y_test - y_test_pred), 'r', label = 'Prediction error')
ax2.legend()
plt.title('Prediction error')
plt.xlabel('Date')
plt.ylabel('Error')
plt.tight_layout()
plt.show()