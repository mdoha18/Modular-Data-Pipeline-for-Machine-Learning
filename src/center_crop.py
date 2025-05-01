from preprocessing import PreprocessingTechnique


class CenterCrop(PreprocessingTechnique):

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image):
        h, w = image.shape[:2]
        top = (h - self.height) // 2
        left = (w - self.width) // 2
        return image[top: top + self.height, left: left + self.width]