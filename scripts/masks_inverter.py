import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt

def invert_masks(path='J:\\Irregular_masks_dataset\\test'):
    example_path = '..\\examples\\celeba\\masks\\celeba_01.png'
    assert os.path.exists(path)
    assert os.path.exists(example_path)
    example_arr = np.array(Image.open(example_path))

    for filename in tqdm(os.listdir(path)):
        full_filename = os.path.join(path, filename)
        img = Image.open(full_filename)
        img_arr = np.array(img)

        if np.mean(img_arr) < 2:
            if img_arr.max() == 255:
                img_arr = 255 - img_arr.astype(int)
            elif img_arr.max() == 1:
                img_arr = 1 - img_arr.astype(int)
                img_arr = img_arr * 255
            else:
                raise NotImplementedError

            assert example_arr.max() == img_arr.max()
            assert example_arr.min() == img_arr.min()

            img = Image.fromarray(np.uint8(img_arr))
            img.save(full_filename)


invert_masks()