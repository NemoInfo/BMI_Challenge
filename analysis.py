import matplotlib.pyplot as plt
from scipy.io import loadmat


def plot_data(trials):
  cmap = plt.get_cmap("twilight", 9)
  _, (a1, a2, a3) = plt.subplots(1, 3, figsize=(10, 3), sharex=True, sharey=True)

  for i in range(trials.shape[0]):
    for d in range(trials.shape[1]):
      a1.plot(*trials[i, d][2][:2, :300], color=cmap(d), linewidth=1, alpha=0.4)
      a2.plot(*trials[i, d][2][:2, 300:-100], color=cmap(d), linewidth=1, alpha=0.4)
      a3.plot(*trials[i, d][2][:2, -100:], color=cmap(d), linewidth=1, alpha=0.4)
  plt.show()


if __name__ == "__main__":
  trials = loadmat("monkeydata_training.mat")['trial']
  plot_data(trials)
