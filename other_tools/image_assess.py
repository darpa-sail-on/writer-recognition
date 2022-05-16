# =============================================================================
# Authors: PAR Government
# Organization: DARPA
#
# Copyright (c) 2019 PAR Government
# All rights reserved.
#==============================================================================


##
## Produce top -K distances between images of single words of hand written text based MSER matched data points
## E.g. compare 'hello' of writer to 'hello' of another
##

import os
import time
import math
import cv2
import numpy as np

def __flannMatcher(d1, d2, args=None):
    FLANN_INDEX_KDTREE = 0
    TREES = 16
    CHECKS = 50
    index_params = {'algorithm':FLANN_INDEX_KDTREE, 'trees':TREES}
    search_params = {'checks':CHECKS}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    return flann.knnMatch(d1, d2, k=2) if d1 is not None and d2 is not None else []

def homographic_distortion(new_src_pts, new_dst_pts):
    try:
        M1, matches = cv2.findHomography(new_src_pts, new_dst_pts, cv2.RANSAC, 4)
    except Exception as ex:
        print ('Fail homography')
        M1 = None
    if M1 is not None:
       return np.max(np.dot( np.ones((3,3)), M1))
    else:
        try:
            M1, matches = cv2.findHomography(new_src_pts, new_dst_pts)
            if M1 is not None:
                return np.max(np.dot(np.ones((3, 3)), M1))
        except Exception as ex:
            print('Fail homography')
            M1 = None
    return 99999

def match_items(item1, item2, matcher=__flannMatcher):
    import functools
    d1 =  item1[1]
    kp1 = item1[0]
    d2 =  item2[1]
    kp2 = item2[0]
    d1 /= (d1.sum(axis=1, keepdims=True) + 1e-7)
    d1 = np.sqrt(d1)

    d2 /= (d2.sum(axis=1, keepdims=True) + 1e-7)
    d2 = np.sqrt(d2)

    try:
      matches = matcher(d1,d2)

      maxmatches = 10000
    # store all the good matches as per Lowe's ratio test.
      good = [m for m, n in matches if m.distance < 0.75 * n.distance]
      good = sorted(good, key=functools.cmp_to_key(lambda g1, g2: -int(max(g1.distance, g2.distance) * 1000)))
      good = good[0:min(maxmatches, len(good))]

      src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
      dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

      return [
            len(good)/float(max(d1.shape[0],d2.shape[0])),
            np.abs(d1.shape[0] - d2.shape[0]),
            homographic_distortion(src_pts, dst_pts),
            np.linalg.norm(src_pts-dst_pts)]
    except:
       return []

def toGray(rgbaimg, type='uint8'):
    a = (rgbaimg[:, :, 3]) if rgbaimg.shape[2] == 4 else np.ones((rgbaimg.shape[0], rgbaimg.shape[1]))
    r, g, b = rgbaimg[:, :, 0], rgbaimg[:, :, 1], rgbaimg[:, :, 2]
    gray = ((0.2126 * r + 0.7152 * g + 0.0722  * b) * a)
    return gray.astype(type)

siftcache = {}
def computeMSER(id, img):
    global siftcache
    if id in siftcache:
        return siftcache[id]
    mser = cv2.MSER_create(_max_area=500)
    regions, boxes = mser.detectRegions(toGray(img))
    mask = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
    for region in regions:
      cv2.fillConvexPoly(mask,region,255)
    detector = cv2.xfeatures2d.SIFT_create(nOctaveLayers=3)#edgeThreshold=10,nOctaveLayers=3,sigma=1.3)
    (kps, descs) = detector.detectAndCompute(toGray(img),mask)
    if descs is None:
       (kps, descs) = detector.detectAndCompute(toGray(img),255*np.ones((img.shape[0],img.shape[1]), dtype=np.uint8))
    siftcache[id] = (kps, descs)
    return (kps, descs)

def evaluate_tansform(item1, item2):
    return [f'{i}' for i in match_items(item1, item2)]

def assess(directory):
  # from imagenet.image_match import ImageSignature
  # gis = ImageSignature(n_levels=5, crop_percentiles=(10, 90))
   map = {}
   for root, dirs, files in os.walk(directory):
        for name in files:
            full = os.path.abspath(os.path.join(root, name))
            if name.endswith('jpg') or name.endswith('png'):
                im = cv2.imread(full)
                if im is None or im.shape[0] == 0:
                    print('bad file %s' % full)
                else:
                    map[name] = im
   done = set()
   with open(os.path.join(directory,'distance.csv'),'w') as fp:
       for iname in map:
            im = computeMSER(iname, map[iname])
            if im is None:
                print (f'skipping {iname}')
            for jname in map:
                print (f'{iname}, {jname}')
                k = (iname, jname) if iname < jname else (jname, iname)
                if iname != jname and k not in done:
                  jm = computeMSER(jname, map[jname])
                  if jm is None:
                      print (f'skipping {jname}')
                  distances = evaluate_tansform(im, jm)
                  d = ','.join(distances)
                  fp.write(f'{iname},{jname},{d}\n')
                  done.add(k)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Dump')
    parser.add_argument('--dir',
                        help='JSON config file')
    args = parser.parse_args()
    assess(args.dir)

if __name__ == '__main__':
    main()

