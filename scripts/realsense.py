# import cv2
# import os
# import time
# import numpy as np

# img_folder = 'img'
# if not os.path.exists(img_folder):
#     os.makedirs(img_folder)

# # Test all video sources, one gives grayscale, other gives color
# cap = cv2.VideoCapture(4)  # Use 1 for /dev/video1
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()

# last_saved_time = time.time()
# last_frame = None

# try:
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Can't receive frame. Exiting ...")
#             break
#         cv2.imshow('frame', frame)
        
#         # Check if the frame has changed
#         if last_frame is None or not np.array_equal(frame, last_frame):
#             current_time = time.time()
#             if current_time - last_saved_time > 3:
#                 img_path = os.path.join(img_folder, f'frame_{int(current_time)}.jpg')
#                 cv2.imwrite(img_path, frame)
#                 last_saved_time = current_time
#                 last_frame = frame.copy()

#         if cv2.waitKey(1) == ord('q'):
#             print("Exiting program...")
#             break

# except KeyboardInterrupt:
#     print("Keyboard interrupt detected. Exiting program...")

# finally:
#     cap.release()
#     cv2.destroyAllWindows()
#     print("Resources released and windows closed.")


import cv2
import os
import time
import numpy as np

img_folder = 'img'
if not os.path.exists(img_folder):
    os.makedirs(img_folder)

cap = cv2.VideoCapture(4)  # Use 1 for /dev/video1
if not cap.isOpened():
    print("Cannot open camera")
    exit()

frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break
        cv2.imshow('frame', frame)
        
        key = cv2.waitKey(1)
        if key == ord(' '):  # Check if spacebar is pressed
            frame_count += 1
            img_path = os.path.join(img_folder, f'frame_{frame_count}.jpg')
            cv2.imwrite(img_path, frame)
            print(f"Frame saved: {img_path}")
        
        elif key == ord('q'):
            print("Exiting program...")
            break

except KeyboardInterrupt:
    print("Keyboard interrupt detected. Exiting program...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released and windows closed.")
