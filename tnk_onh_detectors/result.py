from collections import defaultdict
import json
import os

import numpy as np
from PIL import Image


class DataWithTimestamp:
    def __init__(self, plain_data=None, image_data=None) -> None:
        if plain_data is None:
            self.plain_data = defaultdict(list)
        else:
            self.plain_data = defaultdict(list, plain_data)

        if image_data is None:
            self.image_data = defaultdict(list)
        else:
            self.image_data = defaultdict(list, image_data)

    @classmethod
    def load_from_path(self, output_path):
        name_timestamps = json.load(open(os.path.join(output_path, "name_timestamps.json"), "r"))
        image_data = defaultdict(list)
        for name, timestamp in name_timestamps:
            image = np.array(Image.open(os.path.join(output_path, f"{name}_{timestamp}.png")))
            image_data[name].append((timestamp, image))

        plain_data = json.load(open(os.path.join(output_path, "plain_data.json")))

        return DataWithTimestamp(plain_data, image_data)

    def add_data(self, name, timestamp, data):
        self.plain_data[name].append((timestamp, data))

    def add_image(self, name, timestamp, image):
        self.image_data[name].append((timestamp, image))

    def add_opencv_image(self, name, timestamp, image):
        self.image_data[name].append((timestamp, image[:,:,::-1]))

    def get_images_with_timestamp(self, name):
        return self.image_data[name]

    def get_data(self, name):
        return self.plain_data[name]

    def save(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        json.dump(dict(self.plain_data), open(os.path.join(output_path, "plain_data.json"), "w"))
        name_timestamps = []
        for name, image_tuples in self.image_data.items():
            for timestamp, image in image_tuples:
                name_timestamps.append((name, timestamp))
                Image.fromarray(image).save(os.path.join(output_path, f"{name}_{timestamp}.png"))

        json.dump(name_timestamps, open(os.path.join(output_path, "name_timestamps.json"), "w"))
