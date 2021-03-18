from PIL import Image
from random import *
import numpy as np

def prep_sample(image_file_name, batch_target, images, targets,factor = 0.5):
    """
    Get a list of slices for the provide image:
    generates random crops from each line
    @param image_file_name - path name of image file
    @param batch_target  - targert class
    @param images - add to provided list of images
    @param targets - add to proviced list of targets
    """
    im = Image.open(image_file_name)
    im = im.convert('L') 
    cur_width = im.size[0]
    cur_height = im.size[1]

                # print(cur_width, cur_height)
    height_fac = 113 / cur_height

    new_width = int(cur_width * height_fac)
    size = new_width, 113

    imresize = im.resize((size), Image.ANTIALIAS)  # Resize so height = 113 while keeping aspect ratio
    now_width = imresize.size[0]
    now_height = imresize.size[1]
                # Generate crops of size 113x113 from this resized image and keep random 10% of crops

    avail_x_points = list(range(0, now_width - 113 ))# total x start points are from 0 to width -113

                # Pick random x%
    pick_num = int(len(avail_x_points)*factor)

                # Now pick
    random_startx = sample(avail_x_points,  pick_num)
    
    im = np.asarray(imresize)
#    im[im>50] = 255
#    im[im<=50] = 0

    for start in random_startx:
            images.append(im[:, start:start+113])
            targets.append(batch_target)  


