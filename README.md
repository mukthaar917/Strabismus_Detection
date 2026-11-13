Strabismus Detection & Classification PipelineAn automated Python-based computer vision pipeline designed to detect and classify strabismus using MediaPipe facial landmark analysis and Corneal Light Reflex (CLR) localization. The system computes precise horizontal ($Dx$) and vertical ($Dy$) pupillary asymmetry metrics without requiring manual coordinate priors or hardcoded filename lists.FeaturesFully Automated Landmark Extraction: Leverages MediaPipe Face Mesh (with refine_landmarks=True) to precisely locate iris centers and pupil coordinates.Corneal Light Reflex (CLR) Analysis: Automatically isolates eye regions of interest (ROI) and tracks light reflex centers using OpenCV intensity mapping.Dynamic Ground-Truth Integration: Seamlessly integrates with labels.csv during batch evaluation while maintaining a robust algorithmic fallback threshold for testing new, unseen images.Automated Clinical Reports: Generates structured text reports detailing deviation type (Horizontal vs. Vertical), dominant eye displacement, and clinical subtypes (Esotropia / Exotropia).Project StructureStrabismus_Detection/
│
├── images/                  # Input eye/face images (.jpg, .png, .jpeg)
├── output/                  # Generated locally (results & text reports)
├── labels.csv               # Ground-truth labels for evaluation
├── strabismus_pipeline.py   # Main executable detection script
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
Installation & SetupClone the Repository:git clone https://github.com/mukthaar917/Strabismus_Detection.git
cd Strabismus_Detection
Create and Activate a Virtual Environment:python -m venv .venv
.venv\Scripts\activate
Install Dependencies:pip install -r requirements.txt
UsageOpen strabismus_pipeline.py and set the execution mode to either "single" or "batch":Single Run Mode: Analyzes a specified image file and outputs its annotated result and text report.Batch Evaluation Mode: Iterates through all entries in labels.csv, computes performance metrics, and generates batch reports.Run the script from your terminal:python strabismus_pipeline.py
Sample Detection ResultsInput CaseAnnotated Output & CLR AnalysisClassification StatusCase 1 (1.png)Pupil & CLR MappedSTRABISMUS DETECTED (Esotropia)Case 5 (5.jpg)Normal Aligned MarkersORTHOTROPIC / NEGATIVENote: Full detailed text reports and annotated images are automatically generated in output/reports/ and output/results/ during execution.LicenseDeveloped for computer vision and clinical research applications.
