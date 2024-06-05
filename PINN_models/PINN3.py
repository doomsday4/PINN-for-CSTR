import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

data = pd.read_csv('CSTR_jacket_simData.csv')

X = data[['F2', 'T1', 'CA1', 'F1', 'Fc1', 'Tc1', 'Time', 'V']].values
y = data[['CA', 'T', 'Tc_out']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

#Standardising
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

import numpy as np
import torch

k0 = 1.0e10  # min^-1
E_over_R = 8330.1  # K
rho = 1.0e6  # g/m^3
Cp = 1.0  # cal/(g*K)
delta_Hrxn = -130.0e6  # cal/kmole
a = 1.678e6  # (cal/min)/(K)
b = 0.5

def reactor_equations(state, F, V, Ca0, T0, Fc, Tcin):
    Ca, T = state  # state variables: concentration of A and temperature
    k = k0 * torch.exp(-E_over_R / T)
    
    dCa_dt = (F * (Ca0 - Ca) / V) - (k * Ca)
    dT_dt = ((rho * Cp * F * (T0 - T)) / (rho * Cp * V)) - \
            ((a * Fc**(b+1)) / (Fc + (a * Fc**b) / (2 * rho * Cp * Cp))) * (T - Tcin) + \
            (delta_Hrxn * V * k * Ca / (rho * Cp * V))
    
    return [dCa_dt, dT_dt]


import torch.nn as nn

class PINNModel(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_nodes, output_size):
        super(PINNModel, self).__init__()

        layers = []
        # Input layer
        layers.append(nn.Linear(input_size, hidden_nodes))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_nodes, hidden_nodes))
            layers.append(nn.Tanh())

        # Output layer
        layers.append(nn.Linear(hidden_nodes, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    def pinn_loss(self, batch_X, model_output, batch_y):
        # Inverse transform to get back to original scale
        batch_X = scaler_X.inverse_transform(batch_X.detach().cpu().numpy())
        model_output = scaler_y.inverse_transform(model_output.detach().cpu().numpy())
        batch_y = scaler_y.inverse_transform(batch_y.detach().cpu().numpy())

        # Convert back to tensors
        batch_X = torch.tensor(batch_X, dtype=torch.float32).to(device)
        model_output = torch.tensor(model_output, dtype=torch.float32).to(device)
        batch_y = torch.tensor(batch_y, dtype=torch.float32).to(device)

        # Extract variables
        F, T1, CA1, F1, Fc1, Tc1, Time, V = batch_X.T
        CA, T, Tc_out = model_output.T

        # Calculate derivatives using the reactor equations
        dCa_dt, dT_dt = reactor_equations([CA, T], F, V, CA1, T1, Fc1, Tc1)

        # Calculate time step size
        dt = Time[1:] - Time[:-1]
        dCa_pred = (CA[1:] - CA[:-1]) / dt
        dT_pred = (T[1:] - T[:-1]) / dt

        # Calculate physics-informed loss
        loss_ca = torch.nn.functional.mse_loss(dCa_pred, dCa_dt[:-1])
        loss_t = torch.nn.functional.mse_loss(dT_pred, dT_dt[:-1])
        physics_loss = loss_ca + loss_t

        # Calculate data loss
        data_loss = torch.nn.functional.mse_loss(model_output, batch_y)

        # Total loss
        total_loss = data_loss + 100 * physics_loss

        return total_loss



import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

input_size = X_train_tensor.shape[1]
hidden_layers = 3
hidden_nodes = 128
output_size = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #Currently using CPU only. Using GPU could significanlty improve training time

model = PINNModel(input_size, hidden_layers, hidden_nodes, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

num_epochs = 1000
# batch_size = train_dataset.__sizeof__ #batch descent
# batch size = 1 #stochastic descent
batch_size = 64 #mini_batch descent
train_losses = []
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # Forward pass
        model_output = model(batch_X)

        # Compute losses
        mse_loss = criterion(model_output, batch_y)
        pinn_loss = model.pinn_loss(batch_X, model_output, batch_y)
        loss = mse_loss + pinn_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

model.eval()
with torch.no_grad():
    y_train_pred = model(X_train_tensor.to(device)).cpu().numpy()
    y_train_pred = scaler_y.inverse_transform(y_train_pred)
    y_test_pred = model(X_test_tensor.to(device)).cpu().numpy()
    y_test_pred = scaler_y.inverse_transform(y_test_pred)

    mse_train = mean_squared_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

print(f'Train MSE: {mse_train:.4f}, R2: {r2_train:.4f}')
print(f'Test MSE: {mse_test:.4f}, R2: {r2_test:.4f}')

import matplotlib.pyplot as plt

# Plot of True vs. Predicted values for each output variable
def plot_true_vs_pred(y_true, y_pred, title, ylabel, unit):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='True')
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.title(title)
    plt.xlabel('Time (in seconds)')
    plt.ylabel(ylabel+unit)
    plt.legend()
    plt.show()

# Scatter plot of True vs. Predicted values
def scatter_true_vs_pred(y_true, y_pred, title, unit):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title(title + unit)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.show()

# Plotting for train data
for i, label in enumerate(['CA', 'T', 'Tc_out']):
    if (label == 'CA'):
        unit = ' (in kmole/m^(3))'
    else:
        unit = ' (in Kelvin)'
    plot_true_vs_pred(y_train[:, i], y_train_pred[:, i], f'True vs Predicted for {label} (Train)', label, unit)
    scatter_true_vs_pred(y_train[:, i], y_train_pred[:, i], f'Scatter Plot for {label} (Train)', unit)

# Plotting for test data
for i, label in enumerate(['CA', 'T', 'Tc_out']):
    if (label == 'CA'):
        unit = ' (in kmole/m^(3))'
    else:
        unit = ' (in Kelvin)'
    plot_true_vs_pred(y_test[:, i], y_test_pred[:, i], f'True vs Predicted for {label} (Test)', label, unit)
    scatter_true_vs_pred(y_test[:, i], y_test_pred[:, i], f'Scatter Plot for {label} (Test)', unit)

def plot_learning_curve(train_losses, title='Learning Curve'):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_learning_curve(train_losses)