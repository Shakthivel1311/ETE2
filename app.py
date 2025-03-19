import streamlit as st


try:
    import cv2
except ImportError as e:
    st.error(f"Failed to import OpenCV: {e}")

import pandas as pd
import os
from datetime import datetime
from face_detection import load_known_faces, initialize_attendance_csv, load_attendance_data, detect_faces

# ... rest of your code ...
import cv2
import pandas as pd
import os
from datetime import datetime
from face_detection import load_known_faces, initialize_attendance_csv, load_attendance_data, detect_faces

# Custom CSS to improve UI aesthetics
st.markdown(
    """
    <style>
        .main {
            background-color: #f4f4f4;
        }
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #3498db;
            text-align: center;
        }
        .sidebar .sidebar-content {
            background-color: #2c3e50;
            color: white;
        }
        .stButton > button {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            padding: 10px;
            width: 100%;
            margin: 5px 0;
        }
        .stButton > button:hover {
            background-color: #2980b9;
        }
        .stFileUploader > div > div {
            background-color: #3498db;
            color: white;
        }
        .stRadio > div {
            flex-direction: row;
        }
        .stProgress > div > div > div {
            background-color: #3498db;
        }
        .stExpander > div {
            background-color: #2c3e50;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to display attendance details
def display_attendance_details(csv_file):
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            if not df.empty:
                df["Time"] = pd.to_datetime(df["Time"])
                df["Date"] = df["Time"].dt.date
                df["Time"] = df["Time"].dt.time
                df["Status"] = "Present"

                # Attendance Summary
                st.write("### 📊 Attendance Summary")
                total_present = len(df)
                st.metric("Total Present", total_present)

                # Filter by Date
                st.write("### 📅 Filter by Date")
                selected_date = st.date_input("Select Date", value=datetime.today())
                filtered_df = df[df["Date"] == selected_date]

                if not filtered_df.empty:
                    # Display the filtered table with manual attendance marking
                    st.write("### 🖊️ Mark Attendance Manually")
                    for index, row in filtered_df.iterrows():
                        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                        with col1:
                            st.write(f"**Name:** {row['Name']}")
                        with col2:
                            st.write(f"**Date:** {row['Date']}")
                        with col3:
                            st.write(f"**Time:** {row['Time']}")
                        with col4:
                            status = st.selectbox(
                                "Status",
                                ["Present", "Absent"],
                                index=0 if row["Status"] == "Present" else 1,
                                key=f"status_{index}",
                            )
                            if status != row["Status"]:
                                df.at[index, "Status"] = status
                                df.to_csv(csv_file, index=False)
                                st.success(f"✅ Attendance updated for {row['Name']}.")

                    # Display the filtered table
                    st.write("### 📜 Filtered Attendance Records")
                    st.table(filtered_df[["Date", "Time", "Name", "Status"]])
                else:
                    st.warning(f"No records found for {selected_date}.")

                # Export Attendance
                st.write("### 📤 Export Attendance")
                if st.button("Export to CSV"):
                    df.to_csv("attendance_export.csv", index=False)
                    st.success("Attendance data exported to `attendance_export.csv`.")

                # Full Attendance Records
                st.write("### 📜 Full Attendance Records")
                st.table(df[["Date", "Time", "Name", "Status"]])
            else:
                st.warning("No attendance records found.")
        except Exception as e:
            st.error(f"Error reading attendance file: {e}")
    else:
        st.warning("Attendance file does not exist.")

# Function to add a new student
def add_new_student():
    st.title("📸 Add New Student")
    student_name = st.text_input("Enter Student Name", placeholder="Type student name here...")
    known_faces_dir = "known_faces"
    os.makedirs(known_faces_dir, exist_ok=True)
    option = st.radio("Choose an option", ["Upload Photo", "Capture Photo from Camera"], horizontal=True)
    
    if option == "Upload Photo":
        uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
        if uploaded_file and student_name:
            image_path = os.path.join(known_faces_dir, f"{student_name}.jpg")
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ Photo uploaded and saved for {student_name}.")

    elif option == "Capture Photo from Camera":
        video_capture = cv2.VideoCapture(0)
        stframe = st.empty()
        capture_button = st.button("📷 Capture Photo")
        captured_image = None
        while True:
            ret, frame = video_capture.read()
            if not ret:
                st.error("❌ Failed to capture video.")
                break
            stframe.image(frame, channels="BGR")
            if capture_button:
                captured_image = frame
                break
        video_capture.release()
        if captured_image is not None and student_name:
            image_path = os.path.join(known_faces_dir, f"{student_name}.jpg")
            cv2.imwrite(image_path, captured_image)
            st.success(f"✅ Photo captured and saved for {student_name}.")

# Function to delete a student
def delete_student():
    st.title("🗑️ Delete Student")
    known_faces_dir = "known_faces"
    if os.path.exists(known_faces_dir):
        students = [f.split(".")[0] for f in os.listdir(known_faces_dir) if f.endswith(".jpg")]
        if students:
            selected_student = st.selectbox("Select a student to delete", students)
            if st.button("Delete Student"):
                os.remove(os.path.join(known_faces_dir, f"{selected_student}.jpg"))
                st.success(f"✅ {selected_student} has been deleted.")
        else:
            st.warning("No students found.")
    else:
        st.warning("No students found.")

# Function to capture attendance
def capture_attendance():
    st.title("📷 Capture Attendance")
    known_faces_dir = "known_faces"
    csv_file = "attendance.csv"
    known_face_encodings, known_face_names = load_known_faces(known_faces_dir)
    initialize_attendance_csv(csv_file)
    attendance_data = load_attendance_data(csv_file)
    video_capture = cv2.VideoCapture(0)
    stframe = st.empty()

    # Start and Stop buttons
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("▶️ Start Taking Attendance")
    with col2:
        stop_button = st.button("🛑 Stop Taking Attendance")

    if start_button:
        st.session_state.capture = True

    if stop_button:
        st.session_state.capture = False

    if "capture" not in st.session_state:
        st.session_state.capture = False

    if st.session_state.capture:
        while st.session_state.capture:
            ret, frame = video_capture.read()
            if not ret:
                st.error("❌ Failed to capture video.")
                break
            frame = detect_faces(frame, known_face_encodings, known_face_names, csv_file, attendance_data)
            stframe.image(frame, channels="BGR")
            if not st.session_state.capture:
                break

    video_capture.release()
    cv2.destroyAllWindows()

# Function to view attendance
def view_attendance():
    st.title("📜 View Attendance Details")
    display_attendance_details("attendance.csv")

# Main function
def main():
    st.sidebar.title("📌 Navigation")
    with st.sidebar.expander("Menu", expanded=True):
        if st.button("📷 Capture Attendance"):
            st.session_state.page = "Capture Attendance"
        if st.button("📜 View Attendance"):
            st.session_state.page = "View Attendance"
        if st.button("📸 Add New Student"):
            st.session_state.page = "Add New Student"
        if st.button("🗑️ Delete Student"):
            st.session_state.page = "Delete Student"

    st.sidebar.image("logo.png", width=100)

    if "page" not in st.session_state:
        st.session_state.page = "Capture Attendance"

    if st.session_state.page == "Capture Attendance":
        capture_attendance()
    elif st.session_state.page == "View Attendance":
        view_attendance()
    elif st.session_state.page == "Add New Student":
        add_new_student()
    elif st.session_state.page == "Delete Student":
        delete_student()

if __name__ == "__main__":
    main()