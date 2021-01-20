from collections import deque

import numpy as np

import utils.timestamped_sequence as uts

def from_dict_exp_to_array(dict_exp):
    ua = dict_exp["userAcceleration"]
    return [dict_exp["timestamp"], *ua]

TIMESTAMP_INDEX = 0
Z_INDEX = 2

STATE_NONE = "none"
STATE_GOING_INSIDE = "going_inside"

VALUE_THRESHOLD = 0.35

FRAMERATE = 60

import enum
class DetectorState(enum.Enum):
    Waiting = "waiting"
    Detected = "detected"

class GoingInsideDetector:
    def __init__(self, window_period_s=5.0):
        self.user_accelerations = deque([])
        self.window_period_s = window_period_s
        self.detector_state = DetectorState.Waiting

    def add(self, user_acceleration):
        if len(self.user_accelerations) >= 1 and \
            (user_acceleration[TIMESTAMP_INDEX] \
                - self.user_accelerations[0][TIMESTAMP_INDEX]) \
                    > self.window_period_s:
            self.user_accelerations.popleft()
            
        self.user_accelerations.append(user_acceleration)

    def get_state(self):
        # if len(self.user_accelerations) < self.window_period_s / 2 * FRAMERATE:
        if len(self.user_accelerations) < 20:
            print("too little data")
            return STATE_NONE

        try:
            lowpass_seq = uts.get_lowpass_timestamped_sequence(np.array(self.user_accelerations))
            latest_z = lowpass_seq[-1][Z_INDEX]
            # Means going back.
            print(latest_z)
            if latest_z < 0:
                self.detector_state = DetectorState.Waiting
                return STATE_NONE

            if self.detector_state == DetectorState.Waiting and latest_z > VALUE_THRESHOLD:
                self.detector_state = DetectorState.Detected
                return STATE_GOING_INSIDE
            else:
                return STATE_NONE
        except:
            import IPython; IPython.embed()
