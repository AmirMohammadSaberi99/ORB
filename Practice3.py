import cv2
import time
import matplotlib.pyplot as plt

# 1) Load your two images (change paths as needed)
img1 = cv2.imread('Test.jpg',  cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('Test2.jpg', cv2.IMREAD_GRAYSCALE)
if img1 is None or img2 is None:
    raise FileNotFoundError("One of the input images wasn't found. Check the file paths.")

# 2) Initialize ORB detector and BFMatcher
orb = cv2.ORB_create(
    nfeatures=500,        # max number of keypoints (tweakable)
    scaleFactor=1.2,      # pyramid decimation ratio
    nlevels=8             # number of pyramid levels
)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 3) Detect keypoints & compute descriptors for both images
t0 = time.time()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
t_det = time.time() - t0

# 4) Match descriptors
t0 = time.time()
matches = bf.match(des1, des2)
t_match = time.time() - t0

# 5) Sort matches by descriptor distance (lower = better)
matches = sorted(matches, key=lambda m: m.distance)

# 6) Draw the top N matches
N = 30
img_matches = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    matches[:N],         # take best N
    None,                # output image
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 7) Print out stats
print(f"Detected {len(kp1)} & {len(kp2)} ORB keypoints in {t_det:.3f} s")
print(f"Found {len(matches)} raw matches in {t_match:.3f} s")
print(f"Showing top {N} matches sorted by Hamming distance")

# 8) Display result with Matplotlib
plt.figure(figsize=(12,6))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title(f"ORB Descriptor Matches (top {N})")
plt.axis('off')
plt.tight_layout()
plt.show()
