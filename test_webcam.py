import cv2

# Connect to the IP Webcam stream forwarded by ADB
cap = cv2.VideoCapture("http://127.0.0.1:8080/video")

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to retrieve frame.")
        break

    cv2.imshow("Android Camera Stream", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
