import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1) Load images (grayscale)
# ------------------------------------------------------------------
template = cv2.imread('Test.jpg', cv2.IMREAD_GRAYSCALE)
scene    = cv2.imread('Test2.jpg',    cv2.IMREAD_GRAYSCALE)
if template is None or scene is None:
    raise FileNotFoundError("Check your image paths!")

# ------------------------------------------------------------------
# 2) Detect ORB keypoints + descriptors
# ------------------------------------------------------------------
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(template, None)
kp2, des2 = orb.detectAndCompute(scene,    None)

# ------------------------------------------------------------------
# 3) Match with BFMatcher + ratio test
# ------------------------------------------------------------------
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
# find the two best matches for each descriptor
raw_matches = bf.knnMatch(des1, des2, k=2)

# apply Lowe's ratio test
good = []
for m,n in raw_matches:
    if m.distance < 0.9 * n.distance:
        good.append(m)

print(f"Found {len(good)} good matches")

# ------------------------------------------------------------------
# 4) Estimate homography if we have enough matches
# ------------------------------------------------------------------
MIN_MATCH_COUNT = 10
if len(good) >= MIN_MATCH_COUNT:
    # Source points (in template), dst points (in scene)
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    # Compute homography with RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    # ------------------------------------------------------------------
    # 5) Warp the template corners to the scene
    # ------------------------------------------------------------------
    h, w = template.shape
    template_corners = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    scene_corners    = cv2.perspectiveTransform(template_corners, M)

    scene_color = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    # draw a polygon around the detected template
    cv2.polylines(scene_color, [np.int32(scene_corners)], True, (0,255,0), 3, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # 6) Draw matches + inliers
    # ------------------------------------------------------------------
    result_img = cv2.drawMatches(
        template, kp1,
        scene_color, kp2,
        good, None,
        matchColor=(0,255,0),      # inlier matches in green
        singlePointColor=(255,0,0),
        matchesMask=matchesMask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # ------------------------------------------------------------------
    # 7) Show final result
    # ------------------------------------------------------------------
    plt.figure(figsize=(14,7))
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Template detected with {int(mask.sum())}/{len(good)} inlier matches")
    plt.axis('off')
    plt.show()

else:
    print(f"Not enough matches ({len(good)}/{MIN_MATCH_COUNT}) to compute homography.")
