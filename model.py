import numpy as np
from scipy.io import loadmat
from scipy.signal import convolve2d
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim


class RNN(nn.Module):

  def __init__(self, num_neurons, hidden_sizes=[256, 128, 64]):
    super(RNN, self).__init__()
    sizes = [num_neurons] + hidden_sizes + [2]

    self.fc = nn.ModuleList([nn.Linear(in_size, out_size) for in_size, out_size in zip(sizes[:-1], sizes[1:])])
    self.relu = nn.ReLU()

  def forward(self, spikes):
    x = spikes  # (batch_size, num_neurons)

    for ff in self.fc[:-1]:
      x = self.relu(ff(x))

    return self.fc[-1](x)


class MonkeyDataset(torch.utils.data.Dataset):

  def __init__(self, data, device):
    self.samples = []
    for i_d in np.ndindex(data.shape):
      for t in range(data[i_d][1].shape[0]):
        spikes = data[i_d][1][:, t]  # (num_neurons,)
        pos = data[i_d][2][:, t]  # (2,)

        self.samples.append({
            "inputs": torch.from_numpy(spikes).to(device),
            "target": torch.from_numpy(pos).to(device),
        })

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    return self.samples[idx]


def _model_train(model, dataset, epochs=50, batch_size=64, lr=1e-3):
  model.train()

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
  optimizer = optim.Adam(model.parameters(), lr=lr)
  criterion = nn.MSELoss()
  best_loss = float('inf')

  for epoch in range(epochs):
    total_loss = 0.0
    for batch in dataloader:
      spikes = batch["inputs"]
      targets = batch["target"]

      optimizer.zero_grad()
      outputs = model(spikes)

      loss = criterion(outputs, targets)
      loss.backward()
      optimizer.step()

      total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
      best_loss = avg_loss
      torch.save(model.state_dict(), "best_model.pth")


def model_train(train, use_saved=False):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  state = {"bin": 300, "t0": 0, "device": device}
  Tr, D = train.shape
  N = train[0, 0][1].shape[0]

  if use_saved:
    model = RNN(N).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    return state, model

  # Pre-process data
  for i in tqdm(range(Tr)):
    for d in range(D):
      train[i, d][1] = train_to_rate(train[i, d][1], state["bin"]).astype(np.float32)
      train[i, d][2] = train[i, d][2][:2, state["bin"] - 1:].astype(np.float32)

  dataset = MonkeyDataset(train, device)

  model = RNN(N).to(device)
  _model_train(model, dataset)
  model.eval()

  return state, model


def model_infer(trial, state_model):
  state, model = state_model

  if not trial["prev_hand_pos"]:
    state["t0"] = 0


#     start_pos = torch.from_numpy(trial["start_hand_pos"].astype(np.float32).T).to(state["device"])[None, :]
#   else:
#     start_pos = torch.from_numpy(trial["prev_hand_pos"][-1].astype(np.float32).T).to(state["device"])[None, :]

  T = trial["spikes"].shape[1]
  spikes = torch.from_numpy(train_to_rate(trial["spikes"][:,-state["bin"]:],
                                          state["bin"]).astype(np.float32)).to(state["device"])[None, :, -1]
  outputs = model(spikes)

  state["t0"] = T - state["bin"]
  return outputs[0].cpu().detach().numpy(), (state, model)


def train_to_rate(train, _bin):
  kernel = np.ones((1, _bin)) / _bin
  return convolve2d(train, kernel, mode="valid")


if __name__ == "__main__":
  trials = loadmat("monkeydata_training.mat")['trial']
  model_train(trials[:80])
