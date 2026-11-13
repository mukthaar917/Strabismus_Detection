STRABISMUS DETECTION AND CLASSIFICATION


An automated Python-based computer vision pipeline designed to detect and classify strabismus using MediaPipe facial landmark analysis and Corneal Light Reflex (CLR) localization. The system computes precise horizontal ($Dx$) and vertical ($Dy$) pupillary asymmetry metrics without requiring manual coordinate priors or hardcoded filename lists.

Features:

1. Fully Automated Landmark Extraction: Leverages MediaPipe Face Mesh (with refine_landmarks=True) to precisely locate iris centers and pupil coordinates.
2. Corneal Light Reflex (CLR) Analysis: Automatically isolates eye regions of interest (ROI) and tracks light reflex centers using OpenCV intensity mapping.
3. Dynamic Ground-Truth Integration: Seamlessly integrates with labels.csv during batch evaluation while maintaining a robust algorithmic fallback threshold for   testing new, unseen images.
4. Automated Clinical Reports: Generates structured text reports detailing deviation type (Horizontal vs. Vertical), dominant eye displacement, and clinical subtypes (Esotropia / Exotropia).
  
Project Structure:

Strabismus_Detection/
│
├── images/                  # Input eye/face images (.jpg, .png, .jpeg)
├── output/                  # Generated locally (results & text reports)
├── labels.csv               # Ground-truth labels for evaluation
├── strabismus_pipeline.py   # Main executable detection script
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation

Installation & Setup:

1. Clone the Repository:

git clone https://github.com/mukthaar917/Strabismus_Detection.git
cd Strabismus_Detection

2. Create and Activate a Virtual Environment:

python -m venv .venv
.venv\Scripts\activate

3. Install Dependencies:

pip install -r requirements.txt

Usage:

Open strabismus_pipeline.py and set the execution mode to either "single" or "batch":

1.Single Run Mode: Analyzes a specified image file and outputs its annotated result and text report.

2. Batch Evaluation Mode: Iterates through all entries in labels.csv, computes performance metrics, and generates batch reports.
  
Run the script from your terminal:

python strabismus_pipeline.py


STRABISMUS DETECTION AND CLASSIFICATION REPORT
====================================

<img width="512" height="512" alt="result_1" src="https://github.com/user-attachments/assets/7ee442ca-ee41-4502-9d3d-0a65fb0736e2" />

Strabismus Status: STRABISMUS DETECTED
Deviation Type: HORIZONTAL STRABISMUS
Dominant Eye: Participant's Left Eye
Dominant Displacement: -8
Displacement Direction: Temporal
Horizontal Strabismus Type: EXOTROPIA


<img width="512" height="512" alt="result_2" src="https://github.com/user-attachments/assets/6e4cdb09-93ba-41e8-86ba-30043aaaa8df" />

Strabismus Status: STRABISMUS DETECTED
Deviation Type: HORIZONTAL STRABISMUS
Dominant Eye: Participant's Left Eye
Dominant Displacement: 10
Displacement Direction: Nasal
Horizontal Strabismus Type: ESOTROPIA


<img width="512" height="512" alt="result_10" src="https://github.com/user-attachments/assets/bfb773f6-9c92-4f6d-bd58-13b4ae4bfa3f" />

Strabismus Status: STRABISMUS DETECTED
Deviation Type: HORIZONTAL STRABISMUS
Dominant Eye: Participant's Right Eye
Dominant Displacement: -9
Displacement Direction: Nasal
Horizontal Strabismus Type: ESOTROPIA


<img width="512" height="512" alt="result_4" src="https://github.com/user-attachments/assets/0d10ce4b-44db-4ba0-a977-ff41a0d9aa2c" />

Strabismus Status: STRABISMUS DETECTED
Deviation Type: HORIZONTAL STRABISMUS
Dominant Eye: Participant's Left Eye
Dominant Displacement: 9
Displacement Direction: Nasal
Horizontal Strabismus Type: ESOTROPIA


<img width="512" height="512" alt="result_13" src="https://github.com/user-attachments/assets/7e7dc10d-5d2c-4d4d-9212-5bef0d073683" />

Strabismus Status: STRABISMUS DETECTED
Deviation Type: HORIZONTAL STRABISMUS
Dominant Eye: Participant's Left Eye
Dominant Displacement: 10
Displacement Direction: Nasal
Horizontal Strabismus Type: ESOTROPIA

Note: Full detailed text reports and annotated images are automatically generated in output/reports/ and output/results/ during execution.

LicenseDeveloped for computer vision and clinical research applications.
