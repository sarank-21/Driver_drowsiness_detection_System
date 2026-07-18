import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from mediapipe.python.solutions import face_mesh

# =====================================================
# CONFIG
# =====================================================

EYE_MODEL_PATH = "D:\PROJECTS\Capstone_Project_5\Driver_drowsiness_detection_System\cnn_eye_model.pth"
MOUTH_MODEL_PATH = "D:\PROJECTS\Capstone_Project_5\Driver_drowsiness_detection_System\cnn_mouth_model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =====================================================
# STREAMLIT CONFIG
# =====================================================

st.set_page_config(
    page_title="Driver Fatigue Detection",
    layout="wide"
)

if "fatigue_history" not in st.session_state:
    st.session_state.fatigue_history = []

st.title("🚗 Driver Fatigue Detection System")

if st.button("🔄 Reset Fatigue History"):
    st.session_state.fatigue_history = []
    st.rerun()

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.stImage img {
    border-radius: 12px;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# DEFAULT VALUES
# =====================================================

fatigue = "No Analysis"

fatigue_score = 0
fatigue_score_value = 0

eye_pred = "N/A"
mouth_pred = "N/A"

eye_conf = 0.0
mouth_conf = 0.0

# =====================================================
# IMAGE TRANSFORM
# =====================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

# =====================================================
# CNN MODEL
# =====================================================

class DriverCNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                256 * 14 * 14,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                512,
                num_classes
            )
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model(model_path):

    try:

        checkpoint = torch.load(
            model_path,
            map_location=DEVICE
        )

        classes = checkpoint["classes"]

        model = DriverCNN(
            len(classes)
        )

        model.load_state_dict(
            checkpoint["model_state"]
        )

        model.to(DEVICE)
        model.eval()

        return model, classes

    except Exception as e:

        st.error(
            f"Failed to load model:\n{model_path}\n\n{e}"
        )

        st.stop()

# =====================================================
# PREDICT
# =====================================================

def predict(model, classes, image):

    img = Image.fromarray(image)

    img = transform(img)
    img = img.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(img)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        confidence, pred = torch.max(
            probs,
            1
        )

    return (
        classes[pred.item()],
        confidence.item() * 100
    )

# =====================================================
# FACEMESH LANDMARKS
# =====================================================

LEFT_EYE = [
    33, 133, 160, 159, 158,
    157, 173, 144, 145, 153
]

RIGHT_EYE = [
    362, 263, 387, 386, 385,
    384, 398, 373, 374, 380
]

MOUTH = [
    61,146,91,181,84,17,
    314,405,321,375,291,308,
    78,95,88,178,87,14,
    317,402,318,324
]

# =====================================================
# EYE CROP
# =====================================================

def extract_eye_crop(image, landmarks):

    h, w = image.shape[:2]

    xs = []
    ys = []

    for idx in LEFT_EYE:

        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)

        xs.append(x)
        ys.append(y)

    padding = 15

    x_min = max(min(xs) - padding, 0)
    y_min = max(min(ys) - padding, 0)

    x_max = min(max(xs) + padding, w)
    y_max = min(max(ys) + padding, h)

    eye_crop = image[
        y_min:y_max,
        x_min:x_max
    ]

    return eye_crop

# =====================================================
# MOUTH CROP
# =====================================================

def extract_mouth_crop(image, landmarks):

    h, w = image.shape[:2]

    xs = []
    ys = []

    for idx in MOUTH:

        xs.append(
            int(landmarks[idx].x * w)
        )

        ys.append(
            int(landmarks[idx].y * h)
        )

    padding = 25

    x_min = max(min(xs) - padding, 0)
    y_min = max(min(ys) - padding, 0)

    x_max = min(max(xs) + padding, w)
    y_max = min(max(ys) + padding, h)

    return image[
        y_min:y_max,
        x_min:x_max
    ]

# =====================================================
# FATIGUE SCORE
# =====================================================

def get_fatigue_score(
    eye_state,
    eye_conf,
    mouth_state,
    mouth_conf
):
    score = 0

    eye_state = eye_state.strip().lower()
    mouth_state = mouth_state.strip().lower()

    if eye_state == "closed":
        score += eye_conf * 0.8

    if mouth_state == "yawn":
        score += mouth_conf * 0.4

    return min(round(score), 100)

# =====================================================
# FATIGUE LEVEL
# =====================================================

def get_fatigue_level(score):

    if score >= 70:
        return "Severe Fatigue"

    elif score >= 30:
        return "Mild Fatigue"

    else:
        return "Alert"
    

# =====================================================
# FATIGUE SCORE
# =====================================================

def fatigue_level_to_number(level):

    if level == "Alert":
        return 0

    elif level == "Mild Fatigue":
        return 1

    return 2

# =====================================================
# LOAD MODELS
# =====================================================

with st.spinner("Loading models..."):

    eye_model, eye_classes = load_model(
        EYE_MODEL_PATH
    )

    mouth_model, mouth_classes = load_model(
        MOUTH_MODEL_PATH
    )

# =====================================================
# FILE UPLOADER
# =====================================================

uploaded_file = st.file_uploader(
    "📤 Upload Driver Image",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# PROCESS IMAGE
# =====================================================

if uploaded_file:

    try:

        pil_image = Image.open(
            uploaded_file
        ).convert("RGB")

        image = np.array(
            pil_image
        )

    except Exception as e:

        st.error(
            f"Image loading failed:\n{e}"
        )

        st.stop()

    st.markdown("---")
    st.subheader("Uploaded Image")

    image_col1, image_col2, image_col3 = st.columns(
        [1, 3, 1]
    )

    with image_col2:

        st.image(
            image,
            width=600,
            caption="Driver Image"
        )

    with face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True
    ) as mesh:

        results = mesh.process(image)

        if not results.multi_face_landmarks:

            st.error(
                "❌ No face detected."
            )

        else:

            landmarks = (
                results
                .multi_face_landmarks[0]
                .landmark
            )

            eye_crop = extract_eye_crop(
                image,
                landmarks
            )

            mouth_crop = extract_mouth_crop(
                image,
                landmarks
            )

            if eye_crop.size == 0:

                st.error(
                    "Eye region extraction failed."
                )

                st.stop()

            if mouth_crop.size == 0:

                st.error(
                    "Mouth region extraction failed."
                )

                st.stop()

            st.subheader(
                "Extracted Regions"
            )

            crop_col1, crop_col2 = st.columns(
                2
            )

            with crop_col1:

                st.image(
                    eye_crop,
                    width=250,
                    caption="👁 Eye Region"
                )

            with crop_col2:

                st.image(
                    mouth_crop,
                    width=250,
                    caption="👄 Mouth Region"
                )

            try:

                eye_pred, eye_conf = predict(
                    eye_model,
                    eye_classes,
                    eye_crop
                )

                mouth_pred, mouth_conf = predict(
                    mouth_model,
                    mouth_classes,
                    mouth_crop
                )

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )

                st.stop()

            # =====================================================
            # FATIGUE CALCULATION
            # =====================================================

            fatigue_score_value = get_fatigue_score(
                eye_pred,
                eye_conf,
                mouth_pred,
                mouth_conf
            )

            fatigue = get_fatigue_level(
                fatigue_score_value
            )

            history_score = fatigue_level_to_number(
                fatigue
            )

            # =====================================================
            # STORE FATIGUE HISTORY
            # =====================================================

            st.session_state.fatigue_history.append(
                history_score
            )

            if len(st.session_state.fatigue_history) > 20:
                st.session_state.fatigue_history.pop(0)

            st.subheader(
                "Predictions"
            )

            pred_col1, pred_col2 = st.columns(
                2
            )

            with pred_col1:

                st.info(
                    f"""
👁 Eye State

Prediction: {eye_pred}

Confidence: {eye_conf:.2f}%
"""
                )

                st.progress(
                    eye_conf / 100
                )

            with pred_col2:

                st.info(
                    f"""
👄 Mouth State

Prediction: {mouth_pred}

Confidence: {mouth_conf:.2f}%
"""
                )

                st.progress(
                    mouth_conf / 100
                )
# =====================================================
# RESULTS
# =====================================================

st.markdown("---")

st.subheader("📈 Fatigue Progress")

st.progress(
    min(max(fatigue_score_value, 0), 100) / 100
)

st.metric(
    "Fatigue Score",
    f"{fatigue_score_value}%"
)

st.markdown("---")

st.subheader("Fatigue Assessment")

if fatigue == "Alert":

    st.success(
        "🟢 Driver Status: ALERT"
    )

elif fatigue == "Mild Fatigue":

    st.warning(
        "🟡 Driver Status: MILD FATIGUE"
    )

elif fatigue == "Severe Fatigue":

    st.error(
        "🔴 Driver Status: SEVERE FATIGUE"
    )

else:

    st.info(
        "📤 Upload an image to start analysis."
    )

# =====================================================
# FATIGUE PROGRESSION CURVE
# =====================================================

st.markdown("---")
st.subheader("📈 Fatigue Progression Curve")

if len(st.session_state.fatigue_history) > 0:

    df = pd.DataFrame({
        "Prediction No": range(
            1,
            len(st.session_state.fatigue_history) + 1
        ),
        "Fatigue Score":
            st.session_state.fatigue_history
    })

    st.line_chart(
        df.set_index("Prediction No")
    )

    st.markdown("""
### Fatigue Score Reference

| Score | State |
|--------|--------|
| 0 | 🟢 Alert |
| 1 | 🟡 Mild Fatigue |
| 2 | 🔴 Severe Fatigue |
""")

    avg_score = sum(
        st.session_state.fatigue_history
    ) / len(
        st.session_state.fatigue_history
    )

    st.metric(
        "Average Fatigue Score",
        round(avg_score, 2)
    )

    if avg_score < 0.5:
        st.success(
            "Overall Driver Condition: ALERT"
        )

    elif avg_score < 1.5:
        st.warning(
            "Overall Driver Condition: MILD FATIGUE"
        )

    else:
        st.error(
            "Overall Driver Condition: SEVERE FATIGUE"
        )