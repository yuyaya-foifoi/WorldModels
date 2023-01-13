import numpy as np


def preprocess_obs(obs):
    """
    画像の変換. [0, 255] -> [-0.5, 0.5]
    """
    obs = obs.astype(np.float32)
    normalized_obs = obs / 255.0 - 0.5
    return normalized_obs
