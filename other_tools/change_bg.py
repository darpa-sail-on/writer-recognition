#antique paper bg source: https://www.pinterest.com/pin/352125264589647246/
import cv2
import random
import numpy as np
import os
import glob
import datetime
import copy

def change_bg(img_dir, new_bg):
    begin_time = datetime.datetime.now()

    images = glob.glob(os.path.join(img_dir, "**", "*.png"), recursive=True)
    print(f"Image_count:{len(images)}")

    # handle bg_img stuff:
    bg_original = cv2.imread(new_bg)
    b_channel, g_channel, r_channel = cv2.split(bg_original)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    bg_original = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

    for index, img in enumerate(images):

        #load images
        text_img = cv2.imread(img)
        bg_img = copy.deepcopy(bg_original)

        #add alpha channels
        b_channel, g_channel, r_channel = cv2.split(text_img)
        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
        text_img = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

        #clear white from text image
        mask = cv2.inRange(text_img, np.array([255,255,255, 255]), np.array([255,255,255, 255]))
        text_img[mask>0] = (255,255,255, 0)

        #set offsets to random part of background
        x_offset = int(random.random() * (bg_img.shape[1] - text_img.shape[1]))
        y_offset = int(random.random() * (bg_img.shape[0] - text_img.shape[0]))

        #place text onto bg
        for i, y in enumerate(range(y_offset, y_offset+text_img.shape[0])):
            for j, x in enumerate(range(x_offset, x_offset+text_img.shape[1])):
                if text_img[i][j][3] != 0:
                    bg_img[y][x] = text_img[i][j]

        #grayscale
        final = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)

        #crop to line dimensions
        final = final[y_offset:y_offset+text_img.shape[0], x_offset:x_offset+text_img.shape[1]]

        #write to file
        new_loc = img.replace("lines_fixed", "lines_antique")
        path = os.path.dirname(new_loc)
        if not os.path.exists(path):
            os.makedirs(path)
            print(path)
        cv2.imwrite(new_loc, final)
        print(f"{index+1}/{len(images)}")

    end_time = datetime.datetime.now()
    print("Finished!\nProcessing time: " + str(end_time - begin_time))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        help="Directory for where the image files are being held",
        default=".",
        dest="img_dir",
    )
    parser.add_argument(
        "--new_bg",
        help="New background to use",
        default="antique_bg.jpg",
        dest="new_bg",
    )
    args = parser.parse_args()
    change_bg(args.img_dir, args.new_bg)