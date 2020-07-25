import threading
import numpy as np
import pandas as pd

emo_lock = threading.Condition(threading.Lock())
behav_lock = threading.Condition(threading.Lock())

emo_smooth_lock = threading.Condition(threading.Lock())
behav_smooth_lock = threading.Condition(threading.Lock())

emo_assign_lock = threading.Condition(threading.Lock())
behav_assign_lock = threading.Condition(threading.Lock())

# Feature array
emotion_feature_list = []
behavior_feature_list = []

# Smoothing weight array
emotion_smooth_list = []
behavior_smooth_list = []

# Assign Hits weight array
emotion_assign_list = []
behavior_assign_list = []

final_list = []

INPUT_SIZE = 3800

emotion_feature = np.load("/content/drive/My Drive/Dataset/FEATURES/emotion.npy")
behaviour_feature = np.load("/content/drive/My Drive/Dataset/FEATURES/behavior.npy")
emotion_label = np.load("/content/drive/My Drive/Dataset/FEATURES/per_epoch_y_array.npy")
behaviour_label = np.load("/content/drive/My Drive/Dataset/FEATURES/per_epoch_y_array.npy")
threat_label = np.load("/content/drive/My Drive/Dataset/FEATURES/per_epoch_y_array.npy")
# INPUT_SIZE = 50

# emotion_feature = np.random.rand(INPUT_SIZE,10)
# behaviour_feature = np.random.rand(INPUT_SIZE,10)
# emotion_label = np.random.randint(2, size=INPUT_SIZE)
# behaviour_label = np.random.randint(2, size=INPUT_SIZE)
# threat_label = np.random.randint(2, size=INPUT_SIZE)


# data = pd.read_csv("data/zoo-mini.csv")
# label = data.iloc[:,-1].values
# data = data.iloc[:,1:-1].values
#
# emotion_feature = data[:,:8]
# behaviour_feature = data[:,8:]
# emotion_label = label
# behaviour_label = label
# threat_label = label
#
# INPUT_SIZE = data.shape[0]
