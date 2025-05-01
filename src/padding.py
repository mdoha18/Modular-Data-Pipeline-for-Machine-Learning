from preprocessing import PreprocessingTechnique
import cv2


class Padding(PreprocessingTechnique):
    def __init__(self, target_height, target_width, color):
        self.target_height = target_height
        self.target_width = target_width
        self.color = color

    def __call__(self, image):
        h, w = image.shape[:2]
        top_pad = max((self.target_height - h) // 2, 0)
        bottom_pad = max(self.target_height - h - top_pad, 0)
        left_pad = max((self.target_width - w) // 2, 0)
        right_pad = max(self.target_width - w - left_pad,
                        0)
        return cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad,
                                  right_pad, cv2.BORDER_CONSTANT,
                                  value=self.color)