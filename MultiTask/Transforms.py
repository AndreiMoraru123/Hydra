import numpy as np
import cv2


class Scale(object):

    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)  # add mean to image with 3rd color channel
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)  # add std to image with 3rd color channel

    def __call__(self, image):
        image = image.astype(np.float32)
        image = image / 255.0
        image = (image - self.mean) / self.std
        return np.array(image)


class Normalize(object):

    def __call__(self, image):
        image_copy = np.copy(image)

        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        image_copy = image_copy / 255.0

        return image_copy


class Resize(object):

    def __init__(self, output_size=100):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))

        return img


class ToTensor(object):

    def __call__(self, image):
        if len(image.shape) == 2:
            # add color dimension if missing
            image = image.reshape(image.shape[0], image.shape[1], 1)

        # swap color axis
        # numpy convention: H x W x C
        # torch convention: C X H X W
        # open source software ¯\_(ツ)_/¯
        img = image.transpose((2, 0, 1))

        return image
