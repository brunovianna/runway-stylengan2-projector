import os
import random
from tqdm import tqdm
from glob import glob
from PIL import Image

def crop(im, size, crop_method='No Crop'):
    if crop_method == 'No Crop' or im.size[0] == im.size[1]:
        return im.resize(size)

    elif crop_method == 'Center Crop':
        w, h = im.size
        if w > h:
            left = w / 2 - h / 2
            top = 0
            right = left + h
            bottom = h
        else:
            left = 0
            top = h / 2 - w / 2
            right = w
            bottom = top + w
        return im.crop((left, top, right, bottom)).resize(size)

    elif crop_method == 'Random Crop':
        w, h = im.size
        if w > h:
            left = random.randrange(0, w - h)
            top = 0
            right = left + h
            bottom = h
        else:
            left = 0
            top = random.randrange(0, h - w)
            right = w
            bottom = top + w
        return im.crop((left, top, right, bottom)).resize(size)

    raise Exception('Invalid crop method')

def preprocess_images(source, destination, width, height, crop_method):
    for image_file in tqdm(glob(os.path.join(source, '*'))):
        try:
            with Image.open(image_file) as im:
                try:
                    preprocessed = crop(im, (width, height), crop_method=crop_method)
                except:
                    print('Failed to preprocess', image_file)
                    continue
                base, _ = os.path.splitext(os.path.basename(image_file))
                out_path = os.path.join(destination, base + '.jpg')
                try:
                    preprocessed.convert('RGB').save(out_path)
                except:
                    print('Failed to save preprocessed version', image_file)
                    continue
                try:
                    with Image.open(out_path) as result_im:
                        pass
                except:
                    print('Failed to verify, deleting', out_path)
                    os.remove(out_path)
        except Exception as err:
            print(err)
            print('Failed to load', image_file)
            continue