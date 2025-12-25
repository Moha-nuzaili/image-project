import cv2
import numpy as np

WINDOW_NAME = "Auto Face Mosaic (q: quit, r: force redetect, +/-: pixel size, g: toggle blur)"
PIXEL_BLOCK = 18
USE_GAUSSIAN = False

def apply_mosaic(frame, x, y, w, h, block=18, use_gaussian=False):
    h_img, w_img = frame.shape[:2]
    x = max(0, x); y = max(0, y)
    w = max(1, min(w, w_img - x))
    h = max(1, min(h, h_img - y))
    roi = frame[y:y+h, x:x+w]
    if use_gaussian:
        k = max(3, (block // 2) * 2 + 1)
        frame[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (k, k), 0)
        return frame
    if block < 2: block = 2
    small = cv2.resize(roi, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    frame[y:y+h, x:x+w] = mosaic
    return frame

def detect_skin_face_rect(frame_bgr):
    
    img = frame_bgr
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

    
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0.0

    h_img, w_img = img.shape[:2]
    min_area = (w_img * h_img) * 0.01  
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < min_area:
            continue
        aspect = w / float(h)
        if 0.7 <= aspect <= 1.8:
            
            cx = x + w/2; cy = y + h/2
            dist_center = np.hypot(cx - w_img/2, cy - h_img/2)
            score = area - 2.0 * dist_center  
            if score > best_score:
                best_score = score
                best = (x, y, w, h)
    return best  

def init_camshift(frame, rect):
    x, y, w, h = rect
    hsv_roi = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0, 30, 32)), np.array((180, 255, 255)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    track_window = (x, y, w, h)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    return roi_hist, track_window, term_crit

def main():
    global PIXEL_BLOCK, USE_GAUSSIAN

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("❌   cam does not work.")

    cv2.namedWindow(WINDOW_NAME)
    roi_hist = None
    track_window = None
    term_crit = None
    miss_count = 0
    REDTECT_EVERY_N = 15  

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        
        if roi_hist is None or track_window is None:
            rect = detect_skin_face_rect(frame)
            if rect:
                roi_hist, track_window, term_crit = init_camshift(frame, rect)
                miss_count = 0
            else:
                cv2.putText(frame, "Detecting face...", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('g'):
                    USE_GAUSSIAN = not USE_GAUSSIAN
                elif key in (ord('+'), ord('=')):
                    PIXEL_BLOCK = min(80, PIXEL_BLOCK + 2)
                elif key in (ord('-'), ord('_')):
                    PIXEL_BLOCK = max(2, PIXEL_BLOCK - 2)
                continue

        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        ret, track_window = cv2.CamShift(back_proj, track_window, term_crit)
        pts = cv2.boxPoints(ret).astype(int)
        x, y, w, h = cv2.boundingRect(pts)

        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(frame.shape[1], x+w), min(frame.shape[0], y+h)
        bp_roi = back_proj[y0:y1, x0:x1]
        confidence = float(bp_roi.mean()) if bp_roi.size else 0.0

        if confidence < 5.0 or w*h < 500:
            miss_count += 1
        else:
            miss_count = max(0, miss_count - 1)

        if miss_count >= REDTECT_EVERY_N:
            roi_hist = None
            track_window = None
            miss_count = 0
            cv2.putText(frame, "Re-detecting...", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        out = frame.copy()
        out = apply_mosaic(out, x, y, w, h, block=PIXEL_BLOCK, use_gaussian=USE_GAUSSIAN)
        cv2.polylines(out, [pts], True, (0, 255, 0), 2)
        cv2.putText(out, f"{'Gaussian' if USE_GAUSSIAN else 'Mosaic'} | block={PIXEL_BLOCK} | conf={confidence:.1f} | r: redetect, g: mode, +/-: size, q: quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            roi_hist = None
            track_window = None
            miss_count = 0
        elif key == ord('g'):
            USE_GAUSSIAN = not USE_GAUSSIAN
        elif key in (ord('+'), ord('=')):
            PIXEL_BLOCK = min(80, PIXEL_BLOCK + 2)
        elif key in (ord('-'), ord('_')):
            PIXEL_BLOCK = max(2, PIXEL_BLOCK - 2)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
