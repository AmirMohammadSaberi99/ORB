# Feature Detection and Matching with ORB and SIFT

This repository contains **eight** Python projects demonstrating the use of OpenCV's ORB and SIFT algorithms for feature detection, matching, visualization, and tracking. Each example is self-contained and easy to run.

## Table of Contents
1. [ORB vs SIFT Comparison](#1-orb-vs-sift-comparison)
2. [ORB Keypoint Detection](#2-orb-keypoint-detection)
3. [ORB Descriptor Matching](#3-orb-descriptor-matching)
4. [ORB Descriptor Visualization](#4-orb-descriptor-visualization)
5. [Template Matching with ORB](#5-template-matching-with-orb)
6. [Good Match Counting](#6-good-match-counting)
7. [Real-Time ORB Feature Detection](#7-real-time-orb-feature-detection)
8. [ORB Keypoint Tracking with Optical Flow](#8-orb-keypoint-tracking-with-optical-flow)

---

## Prerequisites
- Python 3.7 or higher
- OpenCV (with contrib modules for SIFT):
  ```bash
  pip install opencv-contrib-python
  ```
- NumPy and Matplotlib:
  ```bash
  pip install numpy matplotlib
  ```

---

## Project Structure
```plaintext
├── compare_orb_sift.py          # 1. Speed & quantity comparison between ORB and SIFT
├── detect_orb_keypoints.py     # 2. Detect & draw ORB keypoints on an image
├── match_orb.py                # 3. Match ORB descriptors between two images
├── visualize_descriptors.py    # 4. Visualize matched ORB descriptor bitmaps
├── template_matching.py        # 5. Identify a template inside a scene with ORB + homography
├── count_good_matches.py       # 6. Count good ORB matches via distance threshold
├── realtime_orb.py             # 7. Real-time ORB feature detection on webcam feed
└── track_orb_optical_flow.py   # 8. Track ORB keypoints across frames using optical flow
```

---

## 1. ORB vs SIFT Comparison
**File:** `compare_orb_sift.py`

Compares ORB and SIFT in terms of detection speed and number of keypoints on two sample images. Outputs a table of counts and timings, and displays keypoint overlays.

**Usage:**
```bash
python compare_orb_sift.py
```

---

## 2. ORB Keypoint Detection
**File:** `detect_orb_keypoints.py`

Detects ORB keypoints on a single image, measures detection time, and visualizes them with scale & orientation.

**Usage:**
```bash
python detect_orb_keypoints.py --image path/to/image.jpg
```

---

## 3. ORB Descriptor Matching
**File:** `match_orb.py`

Matches ORB descriptors between two images using a Hamming-based Brute-Force matcher with cross-check. Draws the top matches sorted by distance.

**Usage:**
```bash
python match_orb.py --img1 path/to/img1.jpg --img2 path/to/img2.jpg
```

---

## 4. ORB Descriptor Visualization
**File:** `visualize_descriptors.py`

After matching, unpacks each 32-byte ORB descriptor into a 16×16 bitmap and displays the bitmaps side-by-side for matched keypoints.

**Usage:**
```bash
python visualize_descriptors.py --img1 path/to/img1.jpg --img2 path/to/img2.jpg
```

---

## 5. Template Matching with ORB
**File:** `template_matching.py`

Finds a smaller template image inside a larger scene by matching ORB features and estimating a homography. Draws the detected outline and inlier matches.

**Usage:**
```bash
python template_matching.py --template path/to/template.jpg --scene path/to/scene.jpg
```

---

## 6. Good Match Counting
**File:** `count_good_matches.py`

Counts the number of ORB descriptor matches under a user-defined Hamming distance threshold. Optionally visualizes only the “good” matches.

**Usage:**
```bash
python count_good_matches.py --img1 path/to/img1.jpg --img2 path/to/img2.jpg --threshold 30
```

---

## 7. Real-Time ORB Feature Detection
**File:** `realtime_orb.py`

Captures live video from a webcam, detects ORB keypoints in real time, and overlays keypoint count and detection time on each frame.

**Usage:**
```bash
python realtime_orb.py
```

---

## 8. ORB Keypoint Tracking with Optical Flow
**File:** `track_orb_optical_flow.py`

Initializes ORB keypoints on the first frame of a video feed and tracks them across frames using Lucas–Kanade optical flow. Periodically re-detects ORB points to maintain robustness.

**Usage:**
```bash
python track_orb_optical_flow.py
```

---

## Contribution
Feel free to open issues or submit pull requests. For major changes, please open an issue first to discuss what you’d like to change.

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

