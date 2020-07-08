import cv2


def get_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(frame[:, :, ::-1])
        else:
            break
    return frames


ordinal_frame_labels = [
    ("going_inside", 7),
    ("going_outside", 14),
    ("going_inside", 21),
    ("going_outside", 30),
    ("going_inside", 50),
    ("going_outside", 51),
    ("going_inside", 57),
    ("going_outside", 70),
    ("going_inside", 77),
    ("going_outside", 83),
    ("going_inside", 89),
    ("going_outside", 96),
    ("going_inside", 102),
    ("going_outside", 109),
    ("going_inside", 115),
    ("going_outside", 122),
    ("going_inside", 140),
    ("going_outside", 149),
]
finish_frame_labels = [
    ("going_inside", 3),
    ("going_outside", 7),
    ("going_inside", 9),
    ("going_outside", 12),
    ("going_inside", 14),
    ("going_outside", 17),
    ("going_inside", 19),
    ("going_outside", 22),
    ("going_inside", 25),
    ("going_outside", 27),
    ("going_inside", 29),
    ("going_outside", 32),
    ("going_inside", 34),
    ("going_outside", 37),
    ("going_inside", 40),
    ("going_outside", 42),
    ("going_inside", 44),
    ("going_outside", 47),
    ("going_inside", 49),
    ("going_outside", 52),
    ("going_inside", 54),
    ("going_outside", 57),
    ("going_inside", 59),
    ("going_outside", 62),
    ("going_inside", 64),
    ("going_outside", 67),
    ("going_inside", 69),
    ("going_outside", 76),
    ("going_inside", 80),
    ("end", 149),
]
ordinal_frames = get_frames(
    "/home/yusuke/Downloads/pixiv/クロカジ/クロカジ - ニッカノ・ハメハメ 14 (78122446) .avi")
finish_frames = get_frames(
    "/home/yusuke/Downloads/pixiv/クロカジ/クロカジ - ニッカノ・ハメハメ 14・ドピュル (78157657) .avi")


def get_from_box(box):
    return ((box[2] + box[0])/2 - 250)/50 * 0.05

def get_x_from_box(box):
    return ((box[3] + box[1])/2 - 425)

class SimpleEstimator:
    def __init__(self):
        self.previous_position = None

    def estimate_inside_length(self, detection_list):
        if detection_list is None:
            return self.previous_position
        for detection in detection_list[1]:
            if detection["label"] == "onahole":
                found = True
                # self.previous_position = get_from_box(detection["box"])
                self.previous_position = get_x_from_box(detection["box"])
                break

        if self.previous_position is None:
            return None


        new_val = (self.previous_position / 15.)
        new_val = new_val if new_val <= 1.0 else 1.0
        new_val = new_val if new_val >= -1.0 else -1.0
        return 1. - (new_val * 0.5 + 0.5)

        # SENSITIVITY_FACTOR = 0.025
        # # SENSITIVITY_FACTOR = 0.05
        # new_val = (self.previous_position / SENSITIVITY_FACTOR)
        # new_val = new_val if new_val <= 1.0 else 1.0
        # new_val = new_val if new_val >= -1.0 else -1.0
        # # new_val = new_val if new_val >= 0 else 0
        # # return (1-(new_val * 0.5 + 0.5))
        # return (new_val * 0.5 + 0.5)


class AnimationPosition:
    def __init__(self, index, frames, frame_labels):
        self.index = index
        self.frames = frames
        self.frame_labels = frame_labels


ANIM_TYPE_GOING_INSIDE = "going_inside"
ANIM_TYPE_GOING_OUTSIDE = "going_outside"


class OnahoAnimationController:
    def __init__(self, estimator):
        self.current_animation = AnimationPosition(
            0, ordinal_frames, ordinal_frame_labels)
        self.estimator = estimator
        self.end_index = None

    def __get_current_clip(self):
        current_frame_label = self.current_animation.frame_labels[self.current_animation.index]
        if self.current_animation.index == 0:
            previous_position = -1
        else:
            previous_position = self.current_animation.frame_labels[
                self.current_animation.index - 1][1]

        return current_frame_label[0], previous_position + 1, current_frame_label[1]

    def __next_clip(self):
        if self.current_animation.index == len(self.current_animation.frames) - 1:
            self.current_animation.index = 0
        else:
            self.current_animation.index = (
                self.current_animation.index + 1) % len(self.current_animation.frame_labels)

    def switch_to_finish(self):
        self.current_animation = AnimationPosition(
            len(finish_frame_labels)-2, finish_frames, finish_frame_labels)

    def consume_detection(self, detection):
        inside_length = self.estimator.estimate_inside_length(detection)
        if inside_length is None:
            return self.current_animation.frames[0]

        anim_type, first_frame, last_frame = self.__get_current_clip()
        # print("Animation Status: ")
        # print(anim_type, first_frame, last_frame)
        # print(inside_length)
        # print("=================")
        if anim_type == ANIM_TYPE_GOING_INSIDE:
            current_frame_id = int(
                (last_frame - first_frame) * inside_length + first_frame)
            current_frame = self.current_animation.frames[current_frame_id]
            if inside_length == 1.0:
                self.__next_clip()

        elif anim_type == ANIM_TYPE_GOING_OUTSIDE:
            current_frame_id = int(
                (last_frame - first_frame) * (1 - inside_length) + first_frame)
            current_frame = self.current_animation.frames[current_frame_id]
            if inside_length == 0.0:
                self.__next_clip()
        elif anim_type == "end":
            if self.end_index == None:
                self.end_index = 0.
            elif int(self.end_index) == len(self.current_animation.frames) - 1:
                return_frame = self.current_animation.frames[int(self.end_index)]
                self.end_index = None
                self.current_animation.index = 0
                return return_frame
            else:
                self.end_index += (1.0 / 1.8)
            return self.current_animation.frames[int(self.end_index)]
        else:
            print(f"wrong animation at {last_frame}")

        return current_frame
