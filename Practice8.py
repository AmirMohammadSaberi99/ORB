import cv2
import numpy as np

def main():
    # ——————————————————————————————
    # 1) Video capture
    # ——————————————————————————————
    cap = cv2.VideoCapture(0)  # use your video file path in place of 0 if needed
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # ——————————————————————————————
    # 2) ORB detector & parameters
    # ——————————————————————————————
    orb = cv2.ORB_create(nfeatures=500)
    # Lucas–Kanade optical flow params
    lk_params = dict(winSize  = (15, 15),
                     maxLevel = 3,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                 10, 0.03))

    # ——————————————————————————————
    # 3) Read first frame and detect initial points
    # ——————————————————————————————
    ret, old_frame = cap.read()
    if not ret:
        print("Failed to read first frame")
        return
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # detect ORB keypoints and convert to point array for optical flow
    kp = orb.detect(old_gray, None)
    p0 = np.array([kp.pt for kp in kp], dtype=np.float32).reshape(-1,1,2)

    # mask for drawing tracks
    mask = np.zeros_like(old_frame)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ——————————————————————————————
        # 4) Track points with LK optical flow
        # ——————————————————————————————
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        else:
            good_new = good_old = []

        # draw the tracks
        for new, old in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a),int(b)), (int(c),int(d)), (0,255,0), 2)
            frame = cv2.circle(frame, (int(a),int(b)), 4, (0,0,255), -1)

        img = cv2.add(frame, mask)

        cv2.putText(img, f"Tracked points: {len(good_new)}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255,255,255), 2)

        cv2.imshow('ORB + LK Tracking', img)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # Esc to quit
            break

        # ——————————————————————————————
        # 5) Every 50 frames, re‐detect ORB to add new points
        # ——————————————————————————————
        frame_idx += 1
        if frame_idx % 50 == 0 or len(good_new) < 10:
            kp = orb.detect(frame_gray, None)
            p0 = np.array([k.pt for k in kp], dtype=np.float32).reshape(-1,1,2)
            mask = np.zeros_like(old_frame)
        else:
            # update for next iteration
            p0 = good_new.reshape(-1,1,2)

        old_gray = frame_gray.copy()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
