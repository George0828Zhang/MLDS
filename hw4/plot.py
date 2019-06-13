import matplotlib.pyplot as plt
import pickle

FileName = "./Q_saved_base/4000.pkl"
with open(FileName, 'rb') as f:
    Data = pickle.load(f)

average_reward_hist = Data[2]
plt.xlabel("episode")
plt.ylabel("average clipped reward")
plt.plot(average_reward_hist)