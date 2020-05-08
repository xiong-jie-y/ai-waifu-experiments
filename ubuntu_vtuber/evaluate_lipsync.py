import facelandmark_utils as flu

# def evaluate(lipsync_func):
#     pass
import glob
import numpy as np

def get_features(mouth_shape):
    features = []
    for dataset_file in glob.glob(f"logs/front_mouth_{mouth_shape}/*.json"):
        landmarks = flu.get_landmark_from_face_frames_file(dataset_file)
        if len(landmarks) == 0:
            print("No landmark")
            continue

        for landmark in landmarks:
            if len(landmark) == 21:
                continue

            flu.normalize_lip(landmark)
            features.append(flu.get_lipsync_feature_v1(landmark))
    return np.array(features)

import matplotlib.pyplot as plt

class_ids = []
feature_totals = []

for class_id, mouth_shape in enumerate(["a", "i", "u", "e", "o", "neautral"]):
    features = get_features(mouth_shape)
    plt.scatter(features[:, 0], features[:,1], label=mouth_shape)
    feature_totals.append(features)
    class_ids += [class_id] * len(features)

feature_totals = np.concatenate(feature_totals)
plt.xlabel("width")
plt.ylabel("height")
plt.legend()
plt.show()

from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
reg = model.fit(feature_totals, class_ids)
print(reg.score(feature_totals, class_ids))

import pickle

pickle.dump(model, open("mouth_classification_model.pkl", "wb"))