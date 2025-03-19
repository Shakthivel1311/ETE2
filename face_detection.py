import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime, timedelta
import pandas as pd
from scipy.spatial import distance

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    # Compute the Euclidean distances between the two sets of vertical eye landmarks
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # Compute the Euclidean distance between the horizontal eye landmarks
    C = distance.euclidean(eye[0], eye[3])
    # Calculate the EAR
    ear = (A + B) / (2.0 * C)
    return ear

# Constants for eye blink detection
EYE_AR_THRESH = 0.25  # EAR threshold to consider a blink
EYE_AR_CONSEC_FRAMES = 3  # Number of consecutive frames to detect a blink

# Initialize the blink counter
blink_counter = 0

# Function to load known faces and their encodings
def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:  # Ensure at least one face is found
                encoding = encodings[0]
                known_face_encodings.append(encoding)
                known_face_names.append(os.path.splitext(filename)[0])

    return known_face_encodings, known_face_names

# Function to initialize the attendance CSV file
def initialize_attendance_csv(csv_file):
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w') as f:
            f.write("Name,Time,Last Attendance Time\n")
        print(f"Created new attendance file: {csv_file}")

# Function to load attendance data into a dictionary
def load_attendance_data(csv_file):
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                print("Attendance file is empty.")
                return {}
            attendance_data = df.set_index('Name').to_dict(orient='index')
            print("Loaded attendance data successfully.")
            return attendance_data
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return {}
    print(f"Attendance file does not exist: {csv_file}")
    return {}

# Function to mark attendance in the CSV file
def mark_attendance(name, csv_file, attendance_data):
    current_time = datetime.now()
    last_attendance_time = attendance_data.get(name, {}).get('Last Attendance Time')

    if last_attendance_time:
        last_attendance_time = datetime.strptime(last_attendance_time, '%Y-%m-%d %H:%M:%S')
        if current_time - last_attendance_time < timedelta(hours=1):
            return False  # Attendance already marked within the last hour

    # Update attendance data
    attendance_data[name] = {
        'Time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
        'Last Attendance Time': current_time.strftime('%Y-%m-%d %H:%M:%S')
    }

    # Save updated attendance data to CSV
    df = pd.DataFrame.from_dict(attendance_data, orient='index').reset_index()
    df.rename(columns={'index': 'Name'}, inplace=True)
    df.to_csv(csv_file, index=False)

    return True

# Function to detect faces and mark attendance
def detect_faces(frame, known_face_encodings, known_face_names, csv_file, attendance_data):
    global blink_counter

    # Resize the frame to 1/4 size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR to RGB

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)  # Stricter threshold
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        # Only mark attendance if the confidence is high
        if matches[best_match_index] and face_distances[best_match_index] < 0.4:  # Confidence threshold
            name = known_face_names[best_match_index]

            # Detect eye blinks for liveness detection
            face_landmarks = face_recognition.face_landmarks(rgb_small_frame, [(top, right, bottom, left)])[0]
            left_eye = face_landmarks['left_eye']
            right_eye = face_landmarks['right_eye']

            # Calculate the eye aspect ratio (EAR) for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Check if the EAR is below the blink threshold
            if ear < EYE_AR_THRESH:
                blink_counter += 1
            else:
                if blink_counter >= EYE_AR_CONSEC_FRAMES:
                    # Blink detected, mark attendance
                    if mark_attendance(name, csv_file, attendance_data):
                        print(f"Attendance marked for {name}")
                    else:
                        print(f"Attendance already marked for {name} in the last hour")
                blink_counter = 0

        # Scale the face locations back to the original frame size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame

# Main function to run the application
def main():
    # Directory containing known faces and CSV file for attendance
    known_faces_dir = "known_faces"
    csv_file = "attendance.csv"

    # Load known faces and initialize the attendance CSV file
    known_face_encodings, known_face_names = load_known_faces(known_faces_dir)
    initialize_attendance_csv(csv_file)

    # Load attendance data
    attendance_data = load_attendance_data(csv_file)

    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame from the webcam
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture video.")
            break

        # Detect faces and mark attendance
        frame = detect_faces(frame, known_face_encodings, known_face_names, csv_file, attendance_data)

        # Display the frame in a window
        cv2.imshow("Face Recognition Attendance", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()