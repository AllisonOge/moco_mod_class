# code adapted from https://github.com/facebookresearch/moco-v3/blob/main/moco/loader.py


class MoCoTransform:
    """Take two random transform of one data"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        x1 = self.base_transform1(x)
        x2 = self.base_transform2(x)
        return [x1, x2]
