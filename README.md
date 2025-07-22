Real-Time Facial Structure Measurements
This project uses computer vision to perform real-time facial structure measurements through a webcam. It detects facial landmarks and calculates the distances between key points, converting them from pixels to centimeters. The application overlays these measurements directly onto the live video feed.

Features
Real-Time Facial Landmark Detection: Utilizes the MediaPipe Face Mesh model to identify 478 landmarks on the face.

On-Screen Measurements: Displays key facial dimensions directly on the video feed.

Real-World Unit Conversion: Converts pixel distances to centimeters by using the average interpupillary distance (6.3 cm) as a reference scale.

Visual Feedback: Renders the face mesh tesselation over the detected face for clear visual tracking.

Measurements Calculated
The following facial dimensions are calculated and displayed:

Eye Distance (Outer Edges)

Nose Length

Lip Width

Jaw Width

Forehead to Chin Length
