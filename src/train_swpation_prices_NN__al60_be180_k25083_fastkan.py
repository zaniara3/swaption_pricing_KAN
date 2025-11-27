import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from fastkan import FastKAN as KAN
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# Load data
path_to_data = os.path.join(ROOT, "..", "data", "samples_to_trainNN_al60_be180_k25083_forgrad.pkl")
with open(path_to_data, 'rb') as f:
    data = pickle.load(f)

factors = data['factors']
swaptions_prices = data['swaption_price']
Talphas = data['Talphas'] / 12

network_path_saved = os.path.join(ROOT, "..", "models", "swaption_network_al60_be180_k25083_forgrad.pth")

# Define Swish activation
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# Define MLP with dropout
class MLP(nn.Module):
    def __init__(self, insize, outsize, HLsizes=[]):
        super().__init__()
        self.model = KAN(layers_hidden=[insize] + HLsizes + [outsize],
                         grid_min=-1.5,
                         grid_max=1.5,
                         num_grids=10
                         )
        self.outact = torch.nn.Softplus(beta=1)

    def forward(self, x):
        x = x.clone()
        x[:, 0] = x[:, 0] / 10.0
        x[:, 1:] = x[:, 1:] * 10.0
        x = self.model(x)
        x = self.outact(x)
        return x


# Define loss
def objective_function(pred, target):
    return F.smooth_l1_loss(pred, target)


# Training loop with DataLoader
def train_evaluate_model_epoches(swap_network, optimizer, train_loader, eval_data, eval_targets, nepochs,
                                 scheduler=None):
    best_objective_value = float('inf')
    best_model_state = None
    wait = 0
    patience = 500

    PerfvecIS, PerfvecOOS = [], []

    for epoch in range(nepochs):
        swap_network.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = swap_network(xb)
            loss = objective_function(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        PerfvecIS.append(avg_loss)

        swap_network.eval()
        with torch.no_grad():
            eval_output = swap_network(eval_data)
            loss_eval = objective_function(eval_output, eval_targets).item()
        PerfvecOOS.append(loss_eval)

        if loss_eval < best_objective_value:
            best_objective_value = loss_eval
            wait = 0
            best_model_state = swap_network.state_dict()
            torch.save(best_model_state, network_path_saved)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered!")
                break

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{nepochs} - IS: {avg_loss:.6f} | OOS: {loss_eval:.6f}")

    return torch.tensor(PerfvecIS), torch.tensor(PerfvecOOS)


# Prepare data
Ntrain = 100000
rng = np.random.default_rng(seed=42)
data = np.hstack([Talphas.reshape(-1, 1), factors])
total_samples = data.shape[0]
indices = rng.permutation(total_samples)
train_idx = indices[:Ntrain]
test_idx = indices[Ntrain:]

data_train = data[train_idx]
data_test = data[test_idx]

y_train = swaptions_prices[train_idx]
y_test = swaptions_prices[test_idx]

train_tensor = TensorDataset(torch.tensor(data_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
train_loader = DataLoader(train_tensor, batch_size=500, shuffle=True)
eval_data = torch.tensor(data_test, dtype=torch.float32)
eval_targets = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Setup model and training
input_size = 4
hidden_size = [16, 32, 16]
output_size = 1
learning_rate = 0.001
nepochs = 5000

swaption_network = MLP(input_size, output_size, hidden_size)
optimizer = optim.Adam(swaption_network.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

PerfvecIS, PerfvecOOS = train_evaluate_model_epoches(
    swaption_network, optimizer, train_loader,
    eval_data, eval_targets,
    nepochs, scheduler=scheduler)


