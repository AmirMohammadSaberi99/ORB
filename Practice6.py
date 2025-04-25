import cv2
import matplotlib.pyplot as plt

# ——————————————————————————————
# 1) Load images (grayscale)
# ——————————————————————————————
img1 = cv2.imread('Test.jpg',  cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('Test2.jpg', cv2.IMREAD_GRAYSCALE)
if img1 is None or img2 is None:
    raise FileNotFoundError("One or both image paths are incorrect.")

# ——————————————————————————————
# 2) Detect ORB keypoints & descriptors
# ——————————————————————————————
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# ——————————————————————————————
# 3) Match descriptors with BFMatcher (Hamming)
# ——————————————————————————————
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# ——————————————————————————————
# 4) Count “good” matches by distance threshold
# ——————————————————————————————
DIST_THRESHOLD = 30  # Hamming distance threshold (tweak as needed)
good_matches = [m for m in matches if m.distance <= DIST_THRESHOLD]

print(f"Total matches found: {len(matches)}")
print(f"Good matches (distance ≤ {DIST_THRESHOLD}): {len(good_matches)}")

# ——————————————————————————————
# 5) (Optional) Draw the good matches
# ——————————————————————————————
drawn = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    good_matches, None,
    matchColor=(0,255,0),
    singlePointColor=(0,0,255),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(12,6))
plt.imshow(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
plt.title(f"Good ORB matches (≤ {DIST_THRESHOLD} Hamming)")
plt.axis('off')
plt.tight_layout()
plt.show()
