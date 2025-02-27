import numpy as np
from scipy.io import loadmat
from scipy.signal import convolve2d
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence


class RNN(nn.Module):

  def __init__(self, N, hidden_sizes=[128, 64]):
    super(RNN, self).__init__()
    self.fc1 = nn.Linear(N + 2, hidden_sizes[0])
    self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
    self.fc3 = nn.Linear(hidden_sizes[1], 2)
    self.relu = nn.ReLU()

  def forward(self, spikes, start_pos, teacher_forcing_targets=None, p=0.5, mask=None):
    B, T, _ = spikes.size()
    outputs = []
    pos = start_pos
    if teacher_forcing_targets is not None:
      teacher_mask = (torch.rand(B, T - 1, device=spikes.device) < p) & mask[:, 1:, 0]

    outputs = torch.zeros(B, T, 2, dtype=torch.float32, device=spikes.device)
    for t in range(T):
      if teacher_forcing_targets is not None and t > 0:
        pos = torch.where(teacher_mask[:, t - 1].unsqueeze(1), teacher_forcing_targets[:, t - 1], pos)

      xt = torch.cat([spikes[:, t], pos], dim=1)
      h1 = self.relu(self.fc1(xt))
      h2 = self.relu(self.fc2(h1))
      pred_pos = self.fc3(h2)

      outputs[:, t] = pred_pos
      if teacher_forcing_targets is None:
        pos = pred_pos

    return outputs


def collate_fn(batch):
  inputs = [sample["inputs"] for sample in batch]
  targets = [sample["target"] for sample in batch]
  lengths = torch.tensor([sample["length"] for sample in batch], device=batch[0]["inputs"].device)

  # Pad sequences; results will have shape (batch_size, max_T, feature_dim)
  inputs_padded = pad_sequence(inputs, batch_first=True)
  targets_padded = pad_sequence(targets, batch_first=True)

  mask = torch.arange(inputs_padded.shape[1], device=batch[0]["inputs"].device)[None, :] < lengths[:, None]

  return {
      "inputs": inputs_padded,
      "target": targets_padded,
      "length": lengths,
      "mask": mask,
  }


class MonkeyDataset(torch.utils.data.Dataset):

  def __init__(self, data, device):
    self.samples = []
    for i_d in np.ndindex(data.shape):
      spikes = data[i_d][1].T  # (T, N)
      pos = data[i_d][2].T
      pos_ = np.copy(pos[:-1])  # (T, 2)
      pos = np.copy(pos[1:])  # (T, 2)

      inputs = np.concatenate([spikes, pos_], axis=1)
      self.samples.append({
          "inputs": torch.from_numpy(inputs).to(device),
          "target": torch.from_numpy(pos).to(device),
          "length": spikes.shape[0],
      })

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    return self.samples[idx]


def model_train_(model, dataset, epochs=10, B=1, lr=1e-3):
  model.train()

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=True, collate_fn=collate_fn)
  optimizer = optim.Adam(model.parameters(), lr=lr)
  criterion = nn.MSELoss()
  best_loss = float('inf')

  for epoch in range(epochs):
    total_loss = 0.0
    for batch in dataloader:
      inputs = batch["inputs"]
      targets = batch["target"]
      mask = batch["mask"].unsqueeze(-1)

      spikes = inputs[:, :, :-2]
      start_pos = inputs[:, 0, -2:]

      optimizer.zero_grad()
      outputs = model(spikes, start_pos, targets, mask=mask)

      loss = (criterion(outputs, targets) * mask).sum() / mask.sum()
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
      train[i, d][2] = train[i, d][2][:2, state["bin"] - 2:].astype(np.float32)

  dataset = MonkeyDataset(train, device)

  model = RNN(N).to(device)
  model_train_(model, dataset)
  model.eval()

  return state, model


def model_infer(trial, state_model):
  state, model = state_model

  if not trial["prev_hand_pos"]:
    state["t0"] = 0
    start_pos = torch.from_numpy(trial["start_hand_pos"].astype(np.float32).T).to(state["device"])[None, :]
  else:
    start_pos = torch.from_numpy(trial["prev_hand_pos"][-1].astype(np.float32).T).to(state["device"])[None, :]

  T = trial["spikes"].shape[1]
  spikes = torch.from_numpy(train_to_rate(trial["spikes"][:, state["t0"]:],
                                          state["bin"]).astype(np.float32).T).to(state["device"])[None, :, :]
  outputs = model(spikes, start_pos)

  state["t0"] = T - state["bin"]

  return outputs[0, -1].cpu().detach().numpy(), (state, model)


def train_to_rate(train, _bin):
  kernel = np.ones((1, _bin)) / _bin
  return convolve2d(train, kernel, mode="valid")


if __name__ == "__main__":
  trials = loadmat("monkeydata_training.mat")['trial']
  model_train(trials[:50])
