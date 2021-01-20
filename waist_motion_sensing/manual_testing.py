#%%
import detectors.skinship as dss
detector = dss.GoingInsideDetector()
test_data = {
    "timestamp": 100000,
    "userAcceleration": [100,100,100]
}
detector.add(dss.from_dict_exp_to_array(test_data))

# %%