import cv2
import pyautogui

#safe paths for face and smile detection models
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
smile_cascade_path = cv2.data.haarcascades + 'haarcascade_smile.xml'

#load classifiers
face_cascade = cv2.CascadeClassifier(face_cascade_path)
smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

#verify successful loading
if face_cascade.empty():
    raise IOError(f"❌ Failed to load face cascade at: {face_cascade_path}")
if smile_cascade.empty():
    raise IOError(f"❌ Failed to load smile cascade at: {smile_cascade_path}")

#opening webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("❌ Could not access the camera. Check your permissions or drivers.")

#optional choice(set frame size for better performance)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not captured. Camera might be disconnected.")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    #mouse movement boundary (white rectangle as visual guide)
    cv2.rectangle(frame, (230, 130), (420, 320), (255, 255, 255), 2)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        mouse_x, mouse_y = pyautogui.position()

        #move left
        if x < 230:
            pyautogui.moveTo(mouse_x - 10, mouse_y, 0.1)
        #move right
        if (x + w) > 420:
            pyautogui.moveTo(mouse_x + 10, mouse_y, 0.1)
        #move up
        if y < 130:
            pyautogui.moveTo(mouse_x, mouse_y - 10, 0.1)
        #move down
        if (y + h) > 320:
            pyautogui.moveTo(mouse_x, mouse_y + 10, 0.1)

        #smile detection (inside face ROI for better performance)
        roi_gray = gray[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.8,
            minNeighbors=22,
            minSize=(25, 25)
        )

        #left click if smiling
        if len(smiles) > 0:
            pyautogui.leftClick()


    cv2.imshow("Face Mouse Controller", frame)    # it would display frame

    #press (q) to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("✅ Program exited successfully.")
        break

#release resources
cap.release()
cv2.destroyAllWindows()