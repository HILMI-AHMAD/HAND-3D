"""
RETROLENS - Filter kamera berbasis hand tracking (OpenCV + MediaPipe + NumPy)

Cara pakai:
  - Bentuk "bingkai" dengan dua tangan (jempol & telunjuk kiri-kanan).
    Area di dalam bingkai akan terkena filter, sisanya tetap normal.
  - MODE 2D : bingkai datar (persegi/trapesium) dari ujung jempol + telunjuk kedua tangan.
  - MODE 3D : kotak 3D dari 5 jari terbuka di kedua tangan (kedalaman dari ukuran tangan).

Kontrol:
  Ganti filter  : sentuhkan jempol ke kelingking (tangan mana pun), atau tekan 'n' / 'p'
  Ganti mode    : kepalkan 2 tangan, atau tekan 'c'
  Keluar        : tekan 'q' atau ESC

Jalankan:
  python retrolens.py
"""

import math
import time

import cv2
import mediapipe as mp
import numpy as np

# ------------------------------ KONFIGURASI ------------------------------
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 960, 540

FILTER_NAMES = [
    "dual-tone", "thermal", "sketch", "pixelate", "glitch", "invert",
    "red-channel", "edge", "blur", "cartoon", "rainbow-wave",
]

# ------------------- PENGATURAN KECEPATAN GANTI FILTER -------------------
FILTER_SWITCH_COOLDOWN = 0.8   # detik jeda antar pergantian filter
MODE_TOGGLE_COOLDOWN = 1.2     # detik jeda antar pergantian mode
PINCH_THRESHOLD = 0.35         # jarak jempol-kelingking / ukuran tangan
FIST_HOLD_TIME = 0.4           # detik kepalan harus ditahan

WINDOW_NAME = "RETROLENS Pake Python"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Indeks landmark
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 0, 4, 8, 12, 16, 20
INDEX_MCP, PINKY_MCP = 5, 17
FINGER_TIPS_PIP = [(8, 6), (12, 10), (16, 14), (20, 18)]  # tanpa jempol


# ------------------------------- FILTER ----------------------------------
def f_dual_tone(img, t):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lut = np.zeros((256, 1, 3), np.uint8)
    dark = np.array([90, 20, 40], np.float32)     # BGR ungu tua
    light = np.array([60, 200, 255], np.float32)  # BGR kuning-oranye
    for i in range(256):
        lut[i, 0] = dark + (light - dark) * (i / 255.0)
    return cv2.LUT(cv2.merge([gray, gray, gray]), lut)


def f_thermal(img, t):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def f_sketch(img, t):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def f_pixelate(img, t, block=14):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // block), max(1, h // block)),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def f_glitch(img, t):
    out = img.copy()
    h, w = img.shape[:2]
    shift = 10
    out[:, :, 0] = np.roll(img[:, :, 0], shift, axis=1)    # B geser kanan
    out[:, :, 2] = np.roll(img[:, :, 2], -shift, axis=1)   # R geser kiri
    rng = np.random.default_rng(int(t * 15))
    for _ in range(8):
        y = int(rng.integers(0, h - 12))
        hh = int(rng.integers(4, 12))
        dx = int(rng.integers(-40, 40))
        out[y:y + hh] = np.roll(out[y:y + hh], dx, axis=1)
    return out


def f_invert(img, t):
    return 255 - img


def f_red_channel(img, t):
    out = np.zeros_like(img)
    out[:, :, 2] = img[:, :, 2]
    return out


def f_edge(img, t):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def f_blur(img, t):
    return cv2.GaussianBlur(img, (0, 0), 12)


def f_cartoon(img, t):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 7, 80, 80)
    return cv2.bitwise_and(color, color, mask=edges)


def f_rainbow_wave(img, t):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    wave = (xs * 0.25 + ys * 0.10 + t * 120.0 + 25 * np.sin(xs / 30.0 + t * 4))
    hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.float32) + wave) % 180).astype(np.uint8)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * 1.4 + 40, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


FILTER_FUNCS = {
    "dual-tone": f_dual_tone, "thermal": f_thermal, "sketch": f_sketch,
    "pixelate": f_pixelate, "glitch": f_glitch, "invert": f_invert,
    "red-channel": f_red_channel, "edge": f_edge, "blur": f_blur,
    "cartoon": f_cartoon, "rainbow-wave": f_rainbow_wave,
}


# --------------------------- FUNGSI BANTU --------------------------------
def lm_xy(hand, idx, w, h):
    p = hand.landmark[idx]
    return np.array([p.x * w, p.y * h], np.float32)


def hand_size(hand, w, h):
    """Ukuran tangan = jarak pergelangan ke pangkal jari tengah."""
    return max(1.0, float(np.linalg.norm(lm_xy(hand, WRIST, w, h) - lm_xy(hand, 9, w, h))))


def fingers_extended(hand, w, h):
    """Jumlah jari (selain jempol) yang terbuka."""
    wrist = lm_xy(hand, WRIST, w, h)
    n = 0
    for tip, pip in FINGER_TIPS_PIP:
        if np.linalg.norm(lm_xy(hand, tip, w, h) - wrist) > \
                np.linalg.norm(lm_xy(hand, pip, w, h) - wrist) * 1.05:
            n += 1
    return n


def is_fist(hand, w, h):
    return fingers_extended(hand, w, h) == 0


def is_open_hand(hand, w, h):
    return fingers_extended(hand, w, h) >= 4


def is_pinky_thumb_touch(hand, w, h):
    d = np.linalg.norm(lm_xy(hand, THUMB_TIP, w, h) - lm_xy(hand, PINKY_TIP, w, h))
    return d / hand_size(hand, w, h) < PINCH_THRESHOLD


def composite_polygon(base, filtered, poly, alpha=1.0):
    """Tempel area `filtered` ke `base` hanya di dalam poligon."""
    mask = np.zeros(base.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, poly.astype(np.int32), 255)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    m = (mask.astype(np.float32) / 255.0 * alpha)[:, :, None]
    return (base * (1 - m) + filtered * m).astype(np.uint8)


def order_quad(pts):
    """Urutkan 4 titik mengelilingi pusat agar poligon tidak menyilang."""
    pts = np.array(pts, np.float32)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]


def put_text(img, text, org, color, scale=0.6, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


# ---------------------------- MODE 2D / 3D --------------------------------
def quad_2d(hands, w, h):
    """Bingkai dari jempol + telunjuk kedua tangan."""
    pts = []
    for hd in hands[:2]:
        pts.append(lm_xy(hd, THUMB_TIP, w, h))
        pts.append(lm_xy(hd, INDEX_TIP, w, h))
    return order_quad(pts)


def draw_2d(frame, filtered, hands, w, h):
    if len(hands) < 2:
        return frame
    quad = quad_2d(hands, w, h)
    out = composite_polygon(frame, filtered, quad)
    cv2.polylines(out, [quad.astype(np.int32)], True, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_3d(frame, filtered, hands, w, h):
    """Kotak 3D: muka depan dari ujung jari kedua tangan, muka belakang digeser
    mengikuti ukuran tangan (makin dekat kamera -> makin dalam kotaknya)."""
    if len(hands) < 2:
        return frame
    # 4 sudut muka depan: jempol & kelingking tiap tangan
    front = []
    for hd in hands[:2]:
        front.append(lm_xy(hd, THUMB_TIP, w, h))
        front.append(lm_xy(hd, PINKY_TIP, w, h))
    front = order_quad(front)

    depth = np.mean([hand_size(hd, w, h) for hd in hands[:2]]) * 0.6
    center = front.mean(axis=0)
    back = center + (front - center) * 0.70 + np.array([depth * 0.35, -depth * 0.35], np.float32)

    out = frame.copy()
    # sisi-sisi kotak (lebih gelap) lalu muka belakang lalu muka depan
    for i in range(4):
        j = (i + 1) % 4
        side = np.array([front[i], front[j], back[j], back[i]], np.float32)
        out = composite_polygon(out, (filtered * 0.55).astype(np.uint8), side)
    out = composite_polygon(out, (filtered * 0.8).astype(np.uint8), back)
    out = composite_polygon(out, filtered, front)

    # garis rangka
    for i in range(4):
        j = (i + 1) % 4
        cv2.line(out, tuple(front[i].astype(int)), tuple(front[j].astype(int)), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(out, tuple(back[i].astype(int)), tuple(back[j].astype(int)), (200, 200, 200), 1, cv2.LINE_AA)
        cv2.line(out, tuple(front[i].astype(int)), tuple(back[i].astype(int)), (200, 200, 200), 1, cv2.LINE_AA)
    return out


# ------------------------------- MAIN -------------------------------------
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        print("Kamera tidak bisa dibuka. Coba ubah CAMERA_INDEX (0/1/2).")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    mode = "2D"
    filter_idx = 0
    last_filter_switch = 0.0
    last_mode_toggle = 0.0
    fist_start = None
    t0 = time.time()

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands_model:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            h, w = frame.shape[:2]
            now = time.time()
            t = now - t0

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = hands_model.process(rgb)
            hands = list(result.multi_hand_landmarks) if result.multi_hand_landmarks else []
            hands.sort(key=lambda hd: hd.landmark[WRIST].x)  # kiri -> kanan di layar

            # --- gesture: ganti filter (jempol-kelingking) ---
            if hands and now - last_filter_switch > FILTER_SWITCH_COOLDOWN:
                if any(is_pinky_thumb_touch(hd, w, h) for hd in hands):
                    filter_idx = (filter_idx + 1) % len(FILTER_NAMES)
                    last_filter_switch = now

            # --- gesture: ganti mode (kepal 2 tangan) ---
            if len(hands) == 2 and all(is_fist(hd, w, h) for hd in hands):
                if fist_start is None:
                    fist_start = now
                elif now - fist_start > FIST_HOLD_TIME and now - last_mode_toggle > MODE_TOGGLE_COOLDOWN:
                    mode = "3D" if mode == "2D" else "2D"
                    last_mode_toggle = now
                    fist_start = None
            else:
                fist_start = None

            # --- render filter ---
            name = FILTER_NAMES[filter_idx]
            filtered = FILTER_FUNCS[name](frame, t)

            # 3D butuh 5 jari terbuka di kedua tangan; kalau belum, tampilkan bingkai 2D
            if mode == "3D" and len(hands) == 2 and all(is_open_hand(hd, w, h) for hd in hands):
                out = draw_3d(frame, filtered, hands, w, h)
            elif mode == "3D":
                out = frame.copy()
            else:
                out = draw_2d(frame, filtered, hands, w, h)

            # --- gambar kerangka tangan ---
            for hd in hands:
                mp_draw.draw_landmarks(
                    out, hd, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

            # --- HUD ---
            mode_txt = ("MODE: 3D (Full 5 Jari)" if mode == "3D" else "MODE: PERSEGI 2D")
            put_text(out, f"{mode_txt} [Tekan 'c' / Kepal 2 Tangan]", (12, 28), (0, 255, 255))
            put_text(out, f"FILTER: {name.upper()} [Sentuh Jempol-Kelingking / 'n' / 'p']",
                     (12, 56), (255, 255, 255), 0.55, 1)

            cv2.imshow(WINDOW_NAME, out)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("n"):
                filter_idx = (filter_idx + 1) % len(FILTER_NAMES)
            elif key == ord("p"):
                filter_idx = (filter_idx - 1) % len(FILTER_NAMES)
            elif key == ord("c"):
                mode = "3D" if mode == "2D" else "2D"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()