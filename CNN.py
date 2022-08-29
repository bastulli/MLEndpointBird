import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Imports
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
from torch.utils.data import DataLoader, Dataset  # Has standard datasets we can import in a nice way
from labels import *

class CNN(nn.Module):
    def __init__(self, state_size, num_classes=1, in_channels=2, kernal=5, stride=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=kernal, stride=stride)        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernal, stride=stride)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernal, stride=stride)
                
        def conv2d_size_out(size):
            return (size - (kernal - 1) - 1) // stride  + 1
                  
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_size)//2)//2)//2
        self.linear_input_size = convw * convw * 128 #Make sure its convw x last conv!

        self.fc1 = nn.Linear(self.linear_input_size, 400)
        self.fc2 = nn.Linear(400, num_classes)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.conv1_bn = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
        
class MyDataset(Dataset):
    def __init__(self, data, window):
        self.data = data
        self.window = window
        self.xData = data['image'].values
        self.yData = torch.LongTensor(data['labels'].values.astype('int'))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        target = self.yData[index]
        img_pth = self.xData[index]
        loaded_arr = np.loadtxt(f"{img_pth}")
        data_val = loaded_arr.reshape(
        loaded_arr.shape[0], loaded_arr.shape[1] // self.window, self.window)
        return torch.FloatTensor(data_val.astype('float32')), target

# read csv file to pandas dataframe
df = pd.read_csv('data/DOTUSD/DOTUSD_60.csv', index_col=0, parse_dates=True)
close = df.close.copy()
dailyVol = getDailyVol(close)
print(dailyVol.to_frame())
tEvents = getTEvents(close,h=dailyVol.mean())
tEvents = tEvents[tEvents > dailyVol.index[0]]
# create target series
ptsl = [1,2]
target=dailyVol
# select minRet
minRet = 0.01
# Run in single-threaded mode on Windows
import platform
if platform.system() == "Windows":
    cpus = 1
else:
    cpus = cpu_count() - 1
t1 = addVerticalBarrier(tEvents, close, numDays=1)
events = getEvents(close,tEvents,ptsl,target,minRet,cpus,t1=t1)
labels = getBins(events, close)
clean_labels = dropLabels(labels)
df['labels']= clean_labels['bin']
df['labels'].fillna(0,inplace=True)
df['labels'] = df['labels'] + 1
df.dropna(inplace=True)
# Hyperparameters
in_channel = 2
num_classes = 3
window_size = 64
learning_rate = 0.000001
batch_size = 64
num_epochs = 100

# create the Nerual Network
model = CNN(state_size=window_size, num_classes=num_classes, in_channels=in_channel)

# random tensor used for testing model input output
x = torch.rand(batch_size, in_channel, window_size, window_size)
print(x.shape)
print(model(x).shape)
print(model(x))
print(model)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' Load trained model and test! '''
load = True

# Initialize network
model = CNN(state_size=window_size,num_classes=num_classes,in_channels=in_channel).to(device)

# Loss and optimizer
#criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# send our pandas dataframe with data to our custom data class
train_dataset = MyDataset(df, window_size)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

def save_model(state, filepath):
    print('Saveing checkpoint...')
    torch.save(state, filepath)

def load_model(checkpoint):
    print('Loading checkpoint...')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

if load:
    load_model(torch.load('model/best_model.pth'))
    train_dataset = MyDataset(df, window_size)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)

if not load:
    #Train Network
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            output = model(data)            
            loss = criterion(output, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()
        
            if batch_idx % 25 == 0:
                print('Epoch {}, Batch {}, train Loss: {:.3f}'.format(epoch, batch_idx, loss.item()))
                
            if batch_idx % 500 == 0:
                checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                save_model(checkpoint,'model/best_model.pth')

else:
    results = []
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        output = model(data)
        loss = criterion(output, targets)
        results.append({'labels': float(output.argmax()), 'loss':float(loss.item())})
        if batch_idx % 500 == 0:
            print('Step {}, Loss: {:.3f}'.format(batch_idx, loss.item()))
        
    model_df = pd.DataFrame(results)
    data_df = pd.DataFrame()
    data_df = df[['close','log_returns']].copy()
    print(model_df['labels'].values)
    data_df['labels'] = model_df['labels'].values
    data_df['buy'] = np.where(data_df['labels']==2,data_df['close'],np.nan)
    data_df['sell'] = np.where(data_df['labels']==0,data_df['close'],np.nan)
    data_df['strategy'] = np.where(data_df['labels']==2, data_df['log_returns'], np.nan)
    data_df['strategy'] = np.where(data_df['labels']==0, 0, data_df['strategy'])
    data_df['strategy'].fillna(method='ffill',inplace=True)
    data_df['creturns'] = data_df['log_returns'].cumsum().apply(np.exp)
    data_df['cstrategy'] = data_df['strategy'].cumsum().apply(np.exp)
    data_df[['creturns','cstrategy']].plot()
    plt.legend()
    
    f, ax = plt.subplots(figsize=(11,8))

    data_df['close'].loc['2021':].plot(ax=ax, alpha=.5)
    data_df['buy'].loc['2021':].plot(ax=ax,ls='',marker='^', markersize=7,
                        alpha=0.75, label='upcross', color='g')
    data_df['sell'].loc['2021':].plot(ax=ax,ls='',marker='v', markersize=7, 
                        alpha=0.75, label='downcross', color='r')

    ax.legend()
    
    plt.show()
    
    