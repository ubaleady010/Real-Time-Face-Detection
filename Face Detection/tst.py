# import cv2
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# webcam = cv2.VideoCapture(0)
# while True:
#     _img=webcam.read()
#     gray = cv2.cvtColor(_img,cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.5,4)
#     for (x,y,w,h) in faces:
#         cv2.rectangle(_img, (x,y), (x+w,y+h), (0,255,0),3)
#     cv2.imshow("Face detection",_img)
#     key = cv2.waitKey(10)
#     if key == 27:
#         break
# webcam.release()
# cv2.destroyAllWindows()


# import cv2
# import os

# # Example for webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not access the webcam.")
#     exit()

# ret, _img = cap.read()
# if not ret:
#     print("Error: Failed to capture a frame.")
#     cap.release()
#     exit()

# gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Grayscale Image from Webcam', gray)
# cap.release()

# # Example for image file
# image_path = "path_to_your_image.jpg"
# if not os.path.exists(image_path):
#     print(f"Error: File not found: {image_path}")
# else:
#     _img = cv2.imread(image_path)
#     if _img is None:
#         print(f"Error: Failed to load the image: {image_path}")
#     else:
#         gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
#         cv2.imshow('Grayscale Image from File', gray)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 


# import cv2

# # Load the pre-trained Haar Cascade Classifier for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Initialize the webcam
# cap = cv2.VideoCapture(0)  # 0 for the default camera

# # Check if the webcam is opened successfully
# if not cap.isOpened():
#     print("Error: Could not access the webcam.")
#     exit()

# # Loop to continuously read and display frames
# while True:
#     ret, frame = cap.read()  # Capture a single frame
#     if not ret:
#         print("Error: Failed to capture a frame.")
#         break

#     # Convert the frame to grayscale (Haar Cascade works with grayscale images)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the image
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # Draw green rectangles around detected faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color for rectangle

#     # Display the frame with detected faces
#     cv2.imshow("Live Webcam Feed", frame)

#     # Break the loop if 'Esc' is pressed (key code 27)
#     if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for the 'Esc' key
#         print("Esc key pressed. Exiting...")
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

 


import cv2

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the camera (0 for laptop camera, 1 for external/mobile camera)
cap = cv2.VideoCapture(0)  # Change to 1 if using a mobile camera

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Loop to continuously read and display frames
while True:
    ret, frame = cap.read()  # Capture a single frame
    if not ret:
        print("Error: Failed to capture a frame.")
        break

    # Convert the frame to grayscale (Haar Cascade works with grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw green rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

    # Add text displaying the number of faces detected
    face_count = len(faces)
    text = f"Faces: {face_count}"
    font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
    font_scale = 0.7  # Font size
    font_color = (0, 0, 255)  # Red color
    thickness = 1  # Thinner text
    position = (10, 30)  # Position of the text (x, y)

    # Add the text to the frame
    cv2.putText(frame, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)

    # Display the frame with detected faces and face count
    cv2.imshow("Live Webcam Feed", frame)

    # Break the loop if 'Esc' is pressed (key code 27)
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for the 'Esc' key
        print("Esc key pressed. Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
