import cv2
import time

def main():
    # 1) Open default camera (0). Change the index if you have multiple cameras.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    # 2) Create the ORB detector
    orb = cv2.ORB_create(
        nfeatures=500,         # max number of keypoints per frame
        scaleFactor=1.2,       # image pyramid decimation ratio
        nlevels=8              # number of pyramid levels
    )

    print("Starting video stream. Press Esc to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab a frame.")
            break

        # 3) Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 4) Detect ORB keypoints
        t0 = time.time()
        keypoints = orb.detect(gray, None)
        dt = (time.time() - t0) * 1000  # milliseconds

        # 5) Draw keypoints (size & orientation)
        output = cv2.drawKeypoints(
            frame, keypoints, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            color=(0,255,0)
        )

        # 6) Overlay FPS and keypoint count
        kp_count = len(keypoints)
        cv2.putText(output,
                    f"Keypoints: {kp_count}  Detect time: {dt:.1f} ms",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2)

        # 7) Show the frame
        cv2.imshow('ORB Feature Detection (press Esc to quit)', output)

        # 8) Break on Esc key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
