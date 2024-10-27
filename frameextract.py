import cv2

# Open the video file
video_path = '/home/gkap/Desktop/Heckathon/Set date/240520/240520_063932_064032.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Set the frame position to the 10th frame (0-based index, so 10th frame is frame 9)
frame_number = 9
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

# Read the 10th frame
success, frame = cap.read()

# Check if the frame was successfully extracted
if success:
    # Save the frame as an image file
    cv2.imwrite('frame_10.jpg', frame)
    print("10th frame saved as 'frame_10.jpg'")
else:
    print("Failed to capture the 10th frame.")

# Release the video capture object
cap.release()
