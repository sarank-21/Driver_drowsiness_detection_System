# 🚗 Driver Drowsiness Detection System

> A deep learning-powered fatigue detection system that analyzes driver eye and mouth states in real time using CNN models and MediaPipe FaceMesh, served through an interactive Streamlit dashboard.

---

## 📖 About the Project

The **Driver Drowsiness Detection System** is an end-to-end computer vision pipeline designed to detect driver fatigue before it leads to accidents. It uses MediaPipe FaceMesh to extract precise eye and mouth regions from a driver image, then runs two independently trained CNN classifiers — one for eye state (open/closed) and one for mouth state (yawn/no_yawn) — to compute a weighted fatigue score. The system was developed by comparing a custom CNN against a transfer-learning MobileNetV2 baseline, selecting the best-performing model for each region, and wrapping the full pipeline in a Streamlit dashboard with live fatigue history tracking.

---

## 🛠️ Development Process

### 1. 📁 Dataset Organization
- Started with a raw source directory containing four class folders: `open`, `closed`, `yawn`, and `no_yawn`
- Programmatically split images into two separate domain datasets: **Eye_Dataset** (`open` / `closed`) and **Mouth_Dataset** (`yawn` / `no_yawn`)
- Filtered for valid image extensions (`.jpg`, `.jpeg`, `.png`) during copy to avoid corrupt files

### 2. ✂️ Train / Validation / Test Split
- Applied a **70 / 15 / 15** split (train / val / test) to both Eye_Dataset and Mouth_Dataset using a custom `split_dataset()` function
- Used `random.shuffle()` before slicing to ensure class-balanced randomization
- Verified split counts using `count_images()` across both datasets and all splits

### 3. 🧠 MobileNetV2 Baseline (Transfer Learning)
- Fine-tuned a pretrained `MobileNetV2` (ImageNet weights) for both eye and mouth classification
- Froze all `model.features` parameters and replaced only `model.classifier[1]` with a task-specific linear layer
- Trained with **Adam optimizer** (`lr=0.0001`), `CrossEntropyLoss`, and augmentations: `RandomHorizontalFlip`, `RandomRotation(10)`
- Saved best checkpoint based on highest validation accuracy across 15 epochs

### 4. 🔬 Custom CNN Training (EyeCNN / MouthCNN)
- Built a 4-block convolutional architecture: `Conv2d → ReLU → MaxPool2d` repeated with channel sizes `32 → 64 → 128 → 256`
- Classifier head: `Flatten → Linear(50176, 512) → ReLU → Dropout(0.5) → Linear(512, num_classes)`
- Trained from scratch with **Adam optimizer** (`lr=0.001`) and same augmentation pipeline
- Saved best `EyeCNN` to `cnn_eye_model.pth` and best `MouthCNN` to `cnn_mouth_model.pth`

### 5. 📊 Model Evaluation & Comparison
- Evaluated both MobileNetV2 and CNN on the held-out test set for each region model
- Generated **confusion matrices** (seaborn heatmap) and **classification reports** (precision, recall, F1-score) for both architectures
- Selected the **custom CNN** as the best model based on test accuracy and generalization

**👁️ Eye State Detection**

| Model | Validation Accuracy | Test Accuracy |
|-------|-------------------|---------------|
| MobileNetV2 | 94.50% | 90.83% |
| **CNN** ✅ | **97.71%** | **96.33%** |

**👄 Mouth State Detection**

| Model | Validation Accuracy | Test Accuracy |
|-------|-------------------|---------------|
| MobileNetV2 | 94.93% | 90.37% |
| **CNN** ✅ | **99.08%** | **99.08%** |

> The custom CNN outperformed MobileNetV2 by **+5.5%** on eye test accuracy and **+8.71%** on mouth test accuracy, making it the clear choice for deployment.

### 6. 🏗️ Streamlit Dashboard Development
- Built a wide-layout Streamlit app (`app4.py`) with `@st.cache_resource` for efficient model loading
- Integrated **MediaPipe FaceMesh** (`static_image_mode=True`) to extract eye and mouth landmark crops from any uploaded driver image
- Implemented a dual-signal **fatigue scoring formula**: `score = eye_conf × 0.8 (if closed) + mouth_conf × 0.4 (if yawn)`, capped at 100
- Added session-state fatigue history (rolling 20 predictions) with a line chart progression curve and aggregate condition summary

---

## ✨ Key Features

### 👁️ Dual-Region CNN Classification
Independently trained CNN classifiers for eye state (open/closed) and mouth state (yawn/no_yawn) — enabling fine-grained, region-specific fatigue signals.

### 🗺️ MediaPipe FaceMesh Landmark Extraction
Uses 468-point facial landmark detection to precisely crop eye and mouth regions with configurable padding, ensuring consistent inputs regardless of face size or distance.

### ⚖️ Weighted Fatigue Score
Combines eye closure confidence (×0.8) and yawn confidence (×0.4) into a single 0–100 fatigue score, giving higher weight to the more clinically significant signal (eye closure).

### 🔴🟡🟢 Three-Level Fatigue Assessment
Maps raw scores to **Alert** (<30), **Mild Fatigue** (30–69), and **Severe Fatigue** (≥70) with color-coded Streamlit alerts for instant driver status visibility.

### 📈 Fatigue Progression Curve
Tracks up to 20 consecutive predictions in session state, plots a line chart of fatigue history, and computes an overall average condition for longitudinal monitoring.

### 🤖 MobileNetV2 vs CNN Benchmarking
Full comparative training pipeline — pretrained MobileNetV2 vs. custom CNN — with confusion matrices and classification reports to justify model selection.

### ⚡ Cached Model Loading
`@st.cache_resource` ensures both CNN models are loaded once per session, preventing repeated disk reads and dramatically reducing inference latency.

### 🎨 Custom Streamlit UI
Rounded image cards, structured column layouts, confidence progress bars, and centered image display create a clean, professional dashboard experience.

---

## 🔍 Features (Detailed)

### Region Extraction Pipeline
- `LEFT_EYE` uses landmarks `[33, 133, 160, 159, 158, 157, 173, 144, 145, 153]` with 15px padding
- `MOUTH` uses 22-point landmark polygon with 25px padding
- Both `extract_eye_crop()` and `extract_mouth_crop()` clamp coordinates to image bounds to prevent out-of-bounds slicing

### CNN Architecture (`DriverCNN` / `EyeCNN` / `MouthCNN`)
- 4 convolutional blocks: `Conv2d(3→32)`, `Conv2d(32→64)`, `Conv2d(64→128)`, `Conv2d(128→256)`, each followed by `ReLU + MaxPool2d(2)`
- Classifier: `Flatten → Linear(50176, 512) → ReLU → Dropout(0.5) → Linear(512, num_classes)`
- `224×224` input with `(0.5, 0.5, 0.5)` mean/std normalization

### MobileNetV2 Transfer Learning
- Loaded pretrained ImageNet weights via `models.mobilenet_v2(weights="DEFAULT")`
- Feature extractor fully frozen (`requires_grad = False`)
- Custom head: `nn.Linear(model.last_channel, num_classes)` replacing the default classifier
- Trained with `lr=0.0001` (lower than CNN to preserve pretrained features)

### Prediction & Confidence
- `predict()` runs a single forward pass, applies `torch.softmax`, and returns `(class_name, confidence_pct)`
- Both eye and mouth confidence displayed as text metrics + `st.progress` bars

### Fatigue History & Analytics
- `fatigue_level_to_number()` maps `Alert → 0`, `Mild Fatigue → 1`, `Severe Fatigue → 2`
- Rolling history stored in `st.session_state.fatigue_history` (capped at 20)
- Average fatigue score thresholds: `<0.5 → Alert`, `0.5–1.5 → Mild`, `>1.5 → Severe`

### Model Evaluation (Notebook)
- `confusion_matrix` and `classification_report` from `sklearn.metrics`
- Heatmap rendered with `seaborn` (`cmap='Blues'`, annotated with counts)
- Evaluated independently for eye model and mouth model on their respective test splits

---

## 🧰 Tech Stack

### 🖥️ Frontend / UI
| Library | Role |
|--------|------|
| `streamlit` | Interactive web dashboard, file uploader, session state, metrics, charts |

### 🧠 Machine Learning
| Library | Role |
|--------|------|
| `torch` | Core deep learning framework, model training, inference |
| `torch.nn` | CNN architecture (`Conv2d`, `Linear`, `Dropout`, `ReLU`, `MaxPool2d`) |
| `torch.optim` | Adam optimizer |
| `torchvision.models` | Pretrained MobileNetV2 for transfer learning baseline |
| `torchvision.datasets` | `ImageFolder` for structured dataset loading |
| `torchvision.transforms` | Image augmentation and normalization pipeline |

### 📊 Data Processing & Evaluation
| Library | Role |
|--------|------|
| `numpy` | Array manipulation, image conversion |
| `pandas` | Fatigue history DataFrame for line chart rendering |
| `sklearn.metrics` | `confusion_matrix`, `classification_report` (precision, recall, F1) |
| `PIL (Pillow)` | Image loading, RGB conversion, transform preparation |

### 📈 Visualization
| Library | Role |
|--------|------|
| `matplotlib` | Confusion matrix figure rendering |
| `seaborn` | Heatmap visualization for confusion matrices |

### 🗺️ Computer Vision
| Library | Role |
|--------|------|
| `mediapipe` | FaceMesh 468-point landmark detection for eye/mouth region extraction |

### ⚙️ Data Pipeline
| Library | Role |
|--------|------|
| `os`, `shutil` | Dataset organization — folder creation and image copying |
| `random` | Shuffle before train/val/test split |
| `torch.utils.data.DataLoader` | Batched data loading with shuffle control |

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

### 2. Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Key libraries:
```
torch torchvision
streamlit
mediapipe
Pillow
numpy
pandas
scikit-learn
matplotlib
seaborn
```

### 4. Prepare the Dataset
- Place raw class folders (`open/`, `closed/`, `yawn/`, `no_yawn/`) inside a `Total/` directory
- Run the dataset organization notebook cells to create `Eye_Dataset/` and `Mouth_Dataset/` with `train/val/test` splits

### 5. Train the Models (Optional)
- Run the **MobileNetV2** cells to generate baseline eye and mouth models
- Run the **CNN** cells to train `EyeCNN` → `cnn_eye_model.pth` and `MouthCNN` → `cnn_mouth_model.pth`
- Models are auto-saved on best validation accuracy

### 6. Update Model Paths
In `app4.py`, update the paths to your saved checkpoints:
```python
EYE_MODEL_PATH = "path/to/cnn_eye_model.pth"
MOUTH_MODEL_PATH = "path/to/cnn_mouth_model.pth"
```

### 7. Run the Application
```bash
streamlit run app.py
```

---

## 💡 Use Cases

1. **Fleet Safety Monitoring** — Deploy at logistics or trucking companies to flag drowsy drivers before long-haul trips begin.

2. **Ride-Sharing Driver Fatigue Checks** — Integrate into driver apps to periodically verify alertness during extended shifts.

3. **Research Baseline** — Use the MobileNetV2 vs. CNN comparison pipeline as a reproducible benchmark for eye/mouth state classification tasks.

4. **Driver Training Systems** — Demonstrate the physiological markers of fatigue (eye closure, yawning) as part of safety training curricula.

5. **Embedded Vehicle Systems** — Adapt the FaceMesh + CNN pipeline for real-time webcam inference in ADAS (Advanced Driver Assistance Systems).

6. **Insurance Telematics** — Provide objective fatigue evidence for accident investigation and risk scoring.

---

## 🔮 Future Enhancements

1. **Real-Time Webcam Mode** — Replace the static image uploader with OpenCV-based live video stream inference
2. **EAR / MAR Geometric Scoring** — Complement CNN predictions with Eye Aspect Ratio and Mouth Aspect Ratio heuristics for ensemble confidence
3. **GRAD-CAM Explainability** — Overlay class activation maps on eye/mouth crops to visualize what the CNN learned
4. **Temporal Fatigue Modeling** — Use an LSTM or sliding window over consecutive frames to detect sustained drowsiness patterns
5. **Audio Alerting** — Trigger beep alerts via `playsound` when severe fatigue is detected
6. **Multi-Face Support** — Extend FaceMesh to `max_num_faces > 1` for monitoring multiple drivers or passengers
7. **Mobile Deployment** — Convert models to ONNX or TFLite for on-device inference on dashcam hardware
8. **Automated Reporting** — Generate per-session PDF fatigue reports with timeline charts for fleet managers

---

## 🏗️ How It Works

```
┌─────────────────────────────────────────────────┐
│           Streamlit Dashboard (app.py)          │
│   File Uploader → Image Display → Reset Button  │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│           MediaPipe FaceMesh Layer               │
│  FaceMesh(static_image_mode=True, max_faces=1)  │
│     468-point landmark detection on RGB image   │
└────────┬────────────────────────────┬────────────┘
         │                            │
         ▼                            ▼
┌─────────────────┐        ┌──────────────────────┐
│ extract_eye_    │        │ extract_mouth_crop()  │
│ crop()          │        │ 22-point MOUTH polygon│
│ LEFT_EYE[10pts] │        │ padding=25px          │
│ padding=15px    │        └──────────┬────────────┘
└───────┬─────────┘                   │
        │                             │
        ▼                             ▼
┌──────────────────┐       ┌───────────────────────┐
│  EyeCNN          │       │  MouthCNN             │
│  cnn_eye_model   │       │  cnn_mouth_model      │
│  .pth            │       │  .pth                 │
│  classes:        │       │  classes:             │
│  [open, closed]  │       │  [no_yawn, yawn]      │
└───────┬──────────┘       └───────────┬───────────┘
        │                              │
        └──────────────┬───────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│            get_fatigue_score()                  │
│  score = eye_conf × 0.8  +  mouth_conf × 0.4   │
│  (if closed)                 (if yawn)          │
│  Capped at 100                                  │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│            get_fatigue_level()                  │
│   < 30  → 🟢 Alert                              │
│  30–69  → 🟡 Mild Fatigue                       │
│   ≥ 70  → 🔴 Severe Fatigue                     │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│        Session State Fatigue History            │
│  fatigue_history[] → rolling 20 predictions    │
│  Line chart + Average Score + Overall Condition │
└─────────────────────────────────────────────────┘
```

---

## 📋 Project Overview

The Driver Drowsiness Detection System is a computer vision safety tool that combines geometric facial landmark extraction with custom deep learning classifiers to assess driver fatigue from static images. The pipeline begins with MediaPipe FaceMesh isolating anatomically precise eye and mouth regions, which are then fed independently into two 4-layer CNN models — `EyeCNN` (classifying `open`/`closed`) and `MouthCNN` (classifying `yawn`/`no_yawn`) — both trained from scratch on 224×224 normalized image inputs. A transfer-learning MobileNetV2 baseline was also trained and evaluated using confusion matrices and classification reports before the custom CNN was selected as the final architecture for superior task-specific performance. Fatigue is quantified through a weighted scoring formula that combines eye closure and yawn confidence into a single 0–100 score, mapped to three alert levels, and tracked across sessions via a Streamlit dashboard with real-time progression curves and aggregate condition summaries.

---

⭐ **If you find this project useful, give it a star on GitHub and share your feedback!**
