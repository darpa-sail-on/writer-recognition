import numpy as np
import os
import cv2
import random
import glob
import datetime

def add_noise(img_dir, prb):
    begin_time = datetime.datetime.now()

    images = glob.glob(os.path.join(img_dir, "**", "*.png"), recursive=True)
    print(f"Image_count:{len(images)}")

    input("pause")
    prob = prb
    for index, img in enumerate(images):
        image = cv2.imread(img)
        output = np.zeros(image.shape,np.uint8)
        thres = 1 - prob 
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        new_loc = img.replace("lines_fixed", "lines_noise")
        path = os.path.dirname(new_loc)
        if not os.path.exists(path):
            os.makedirs(path)
            print(path)
        cv2.imwrite(new_loc, output)
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
        "--prob",
        help="Probability used for determining the amount of salt and pepper noise",
        default=.02,
        dest="prob",
    )
    args = parser.parse_args()
    add_noise(args.img_dir, args.prob)
