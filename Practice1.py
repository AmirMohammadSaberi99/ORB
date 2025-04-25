import cv2
import time
import matplotlib.pyplot as plt

# 1) Load your image (change the path as needed)
img_path = 'Test.jpg'
img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img_gray is None:
    raise FileNotFoundError(f"Could not load image at {img_path}")

# 2) Initialize ORB detector
orb = cv2.ORB_create()

# 3) Detect keypoints (and time it)
t0 = time.time()
keypoints = orb.detect(img_gray, None)
elapsed = time.time() - t0

# 4) Draw keypoints (size & orientation)
img_kp = cv2.drawKeypoints(
    img_gray, keypoints, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# 5) Print stats
print(f"Detected {len(keypoints)} ORB keypoints in {elapsed:.4f} seconds")

# 6) Display
plt.figure(figsize=(8,8))
plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
plt.title(f"ORB keypoints ({len(keypoints)} pts)")
plt.axis('off')
plt.show()
