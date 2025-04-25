import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

# ——————————————————————————————
# 1) Load images
# ——————————————————————————————
img1 = cv2.imread('Test.jpg',  cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('Test2.jpg', cv2.IMREAD_GRAYSCALE)
if img1 is None or img2 is None:
    raise FileNotFoundError("Check your image paths!")

# ——————————————————————————————
# 2) Detect ORB keypoints + descriptors
# ——————————————————————————————
orb = cv2.ORB_create(nfeatures=500)
t0 = time.time()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
print(f"ORB detect+compute: {len(kp1)} / {len(kp2)} keypoints  in {time.time() - t0:.3f}s")

# ——————————————————————————————
# 3) Match with Brute-Force (Hamming + crossCheck)
# ——————————————————————————————
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
t0 = time.time()
matches = bf.match(des1, des2)
print(f"Found {len(matches)} raw matches in {time.time() - t0:.3f}s")

# sort by distance (i.e. Hamming)
matches = sorted(matches, key=lambda m: m.distance)

# ——————————————————————————————
# 4) Draw top‐N matches
# ——————————————————————————————
N = 30
img_matches = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    matches[:N], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(12,6))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title(f"Top {N} ORB descriptor matches")
plt.axis('off')
plt.tight_layout()
plt.show()

# ——————————————————————————————
# 5) Visualize the descriptors for each match
#    We take the first M matches to keep the plot readable.
# ——————————————————————————————
M = 10
fig, axes = plt.subplots(M, 2, figsize=(4, 2*M))
fig.suptitle("ORB descriptors as 16×16 bit‐maps\n(left: image1 → right: image2)")

for i, m in enumerate(matches[:M]):
    d1 = des1[m.queryIdx]  # shape (32,) uint8
    d2 = des2[m.trainIdx]

    # unpack bits to shape (256,)
    b1 = np.unpackbits(d1)
    b2 = np.unpackbits(d2)

    # reshape to 16×16 for display
    bmp1 = b1.reshape(16,16)
    bmp2 = b2.reshape(16,16)

    ax1, ax2 = axes[i]
    ax1.imshow(bmp1, cmap='gray', vmin=0, vmax=1)
    ax1.set_title(f"Match {i+1}\nImg1")
    ax1.axis('off')

    ax2.imshow(bmp2, cmap='gray', vmin=0, vmax=1)
    ax2.set_title("Img2")
    ax2.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
