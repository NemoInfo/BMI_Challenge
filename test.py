from model import model_train, model_infer
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np


def test_model():
  trials: np.ndarray = loadmat("monkeydata_training.mat")['trial']

  cmap = plt.get_cmap("twilight", 9)
  np.random.seed(69)
  idx = np.random.permutation(trials.shape[0])
  split = 50

  tr_data = trials[idx[:split]]
  te_data = trials[idx[split:]]

  print("Testing the continuous posiotion estimator ...")

  mean_sq_err = 0
  n_predictions = 0

  model = model_train(tr_data)

  for i in range(te_data.shape[0]):
    print(f"Decoding block {i+1} out of {te_data.shape[0]}")
    for d in range(te_data.shape[1]):
      prev_hand_pos = []
      times = range(320, te_data[i, d][1].shape[1], 20)

      for t in times:
        trial = {
            "id": te_data[i, d][0][0, 0],
            "spikes": te_data[i, d][1],
            "prev_hand_pos": prev_hand_pos,
            "start_hand_pos": te_data[i, d][2][:2, 0],
        }

        decoded_hand_pos, new_model = model_infer(trial, model)
        prev_hand_pos.append(decoded_hand_pos)
        mean_sq_err += np.linalg.norm(decoded_hand_pos - te_data[i, d][2][:2, t])**2
        n_predictions += 1

      plt.plot(*prev_hand_pos, color=cmap(d), linewidth=1, alpha=0.4)
      plt.plot(*te_data[i, d][2][:2, times], color=cmap(d), linewidth=1, alpha=0.4)

  print(mean_sq_err / n_predictions)
  plt.show()


if __name__ == "__main__":
  test_model()
