
import cv2

for i in range(6):

    cap = cv2.VideoCapture(i)

    if cap.isOpened():

        success, frame = cap.read()

        if success:
            print(f"Kamera ditemukan pada indeks: {i}")

            cv2.imshow(f"Kamera Indeks {i}", frame)
            cv2.waitKey(2000)

        cap.release()

cv2.destroyAllWindows()
