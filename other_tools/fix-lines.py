import cv2
import os
import glob
import datetime

def fix_lines(img_dir, dest_dir):
    begin_time = datetime.datetime.now()

    images = glob.glob(os.path.join(img_dir, "**", "*.png"), recursive=True)
    print(f"Image_count:{len(images)}")

    for index, image in enumerate(images):
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th, threshed = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        threshed = cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR)
        threshed_marker = threshed>0
        inverted_color = (img*threshed_marker) + (255-threshed)
        new_loc = os.path.join(dest_dir,image[len(img_dir)+1:])
        path = os.path.dirname(new_loc)
        if not os.path.exists(path):
            os.makedirs(path)
            print(path)
        cv2.imwrite(new_loc, inverted_color)
        print(f"{index}/{len(images)}")

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
        "--destination",
        help="Directory for where the image files are being held",
        default=".",
        dest="dest_dir",
    )
    args = parser.parse_args()
    fix_lines(args.img_dir, args.dest_dir)
