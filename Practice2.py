import cv2
import time
import matplotlib.pyplot as plt
import pandas as pd

# 1) Load images
img_paths = {
    'Stones': 'Test.jpg',
    'Flowers': 'Test2.jpg'
}
images = {name: cv2.imread(path, cv2.IMREAD_GRAYSCALE)
          for name, path in img_paths.items()}

# 2) Initialize detectors
orb  = cv2.ORB_create()
sift = cv2.SIFT_create()

def detect_and_draw(detector, img):
    """Detect keypoints, time the operation, and return keypoints + drawing."""
    t0 = time.time()
    kpts = detector.detect(img, None)
    dt = time.time() - t0
    # draw rich keypoints (size & orientation)
    drawn = cv2.drawKeypoints(img, kpts, None,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return kpts, dt, drawn

# 3) Loop over images and detectors
records = []
overlay_images = []
for name, img in images.items():
    for det_name, det in [('ORB', orb), ('SIFT', sift)]:
        kpts, dt, drawn = detect_and_draw(det, img)
        records.append({
            'Image':    name,
            'Detector': det_name,
            'Keypoints': len(kpts),
            'Time (s)': round(dt, 4)
        })
        overlay_images.append((f"{det_name} – {name}", drawn))

# 4) Print summary table
df = pd.DataFrame(records)
print(df.to_string(index=False))

# 5) Display the keypoint overlays
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.ravel()

for ax, (title, overlay) in zip(axes, overlay_images):
    # Convert BGR->RGB for matplotlib
    ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()
