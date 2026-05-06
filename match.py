import argparse
import numpy as np
import os
import torch
import tqdm
import cv2
import matplotlib.pyplot as plt

from modules.xfeat import XFeat

parser = argparse.ArgumentParser(description="Match images using XFeat and XFeat*")
parser.add_argument('--img1', type=str, default='./assets/frame1.png', help='Path to the first image')
parser.add_argument('--img2', type=str, default='./assets/frame2.png', help='Path to the second image')
parser.add_argument('--output_xfeat', type=str, default='semi_dense_matching_xfeat.png', help='Output file for standard XFeat matching')
parser.add_argument('--output_xfeat_star', type=str, default='semi_dense_matching_xfeat_star.png', help='Output file for XFeat* matching')
args = parser.parse_args()

xfeat = XFeat(ablation_type="skip_default")
im1 = cv2.imread(args.img1)
im2 = cv2.imread(args.img2)

def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    # Calculate homography
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Get corners of the first image
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0],[w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)
    # Warp corners to the second image space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i-1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  #green

    keypoints1 =[cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches =[cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]

    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)

    return img_matches

mkpts_0, mkpts_1 = xfeat.match_xfeat(im1, im2, top_k = 1012)

canvas = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, im2)
plt.figure(figsize=(12,12))
plt.imshow(canvas[..., ::-1])
plt.axis('off')

plt.savefig(
    args.output_xfeat,
    dpi=300,
    bbox_inches='tight',
    pad_inches=0
)
plt.close()

mkpts_0, mkpts_1 = xfeat.match_xfeat_star(im1, im2, top_k = 16000)

canvas = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, im2)
plt.figure(figsize=(12,12))
plt.imshow(canvas[..., ::-1])
plt.axis('off')

plt.savefig(
    args.output_xfeat_star,
    dpi=300,
    bbox_inches='tight',
    pad_inches=0
)
plt.close()