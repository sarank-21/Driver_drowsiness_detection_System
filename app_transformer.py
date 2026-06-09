import streamlit as st
import torch
import pandas as pd
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

from mediapipe.python.solutions import face_mesh

# =====================================================
# CONFIG
# =====================================================

EYE_MODEL_PATH = "D:\PROJECTS\project_no_5\Driver_Drowsiness_Detection\eye_model.pth"
MOUTH_MODEL_PATH = "D:\PROJECTS\project_no_5\Driver_Drowsiness_Detection\mouth_crop_model.pth"

DEVICE = torch.device("cpu")

# =====================================================
# STREAMLIT CONFIG
# =====================================================

st.set_page_config(
    page_title="Driver Fatigue Detection",
    layout='wide'
)

# =====================================================
# SESSION STATE
# =====================================================

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

st.markdown("---")

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
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model(model_path):

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    classes = checkpoint["classes"]

    model = models.mobilenet_v2(weights=None)

    model.classifier[1] = nn.Linear(
        model.last_channel,
        len(classes)
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    return model, classes

# =====================================================
# PREDICT
# =====================================================

def predict(model, classes, image):

    img = Image.fromarray(image)

    img = transform(img).unsqueeze(0)

    with torch.no_grad():

        outputs = model(img)

        _, pred = torch.max(outputs, 1)

    return classes[pred.item()]

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

    for idx in LEFT_EYE + RIGHT_EYE:

        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)

        xs.append(x)
        ys.append(y)

    padding = 20

    x_min = max(min(xs) - padding, 0)
    y_min = max(min(ys) - padding, 0)

    x_max = min(max(xs) + padding, w)
    y_max = min(max(ys) + padding, h)

    return image[y_min:y_max, x_min:x_max]

# =====================================================
# MOUTH CROP
# =====================================================

def extract_mouth_crop(image, landmarks):

    h, w = image.shape[:2]

    xs = []
    ys = []

    for idx in MOUTH:

        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)

        xs.append(x)
        ys.append(y)

    padding = 25

    x_min = max(min(xs) - padding, 0)
    y_min = max(min(ys) - padding, 0)

    x_max = min(max(xs) + padding, w)
    y_max = min(max(ys) + padding, h)

    return image[y_min:y_max, x_min:x_max]

# =====================================================
# DECISION FUSION
# =====================================================

def get_fatigue_level(eye_state, mouth_state):

    eye_state = eye_state.lower()
    mouth_state = mouth_state.lower()

    if eye_state == "closed":
        return "Severe Fatigue"

    elif eye_state == "open" and mouth_state == "yawn":
        return "Mild Fatigue"

    return "Alert"
# =====================================================
# FATIGUE SCORE
# =====================================================

def fatigue_score(level):

    if level == "Alert":
        return 0

    elif level == "Mild Fatigue":
        return 1

    return 2
# =====================================================
# LOAD MODELS
# =====================================================

with st.spinner("Loading models..."):
    eye_model, eye_classes = load_model(EYE_MODEL_PATH)
    mouth_model, mouth_classes = load_model(MOUTH_MODEL_PATH)

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

    pil_image = Image.open(uploaded_file).convert("RGB")

    image = np.array(pil_image)

    st.markdown("### Uploaded Image")

    col1, col2, col3 = st.columns([0.5, 2, 1])

    with col2:
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

            st.error("❌ No face detected in the uploaded image.")

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

            st.markdown("### Extracted Regions")

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    eye_crop,
                    width=250,
                    caption="👁 Eye Region"
                )

            with col2:
                st.image(
                    mouth_crop,
                    width=250,
                    caption="👄 Mouth Region"
                )

            eye_pred = predict(
                eye_model,
                eye_classes,
                eye_crop
            )

            mouth_pred = predict(
                mouth_model,
                mouth_classes,
                mouth_crop
            )

            st.markdown("### Predictions")

            col1, col2 = st.columns(2)

            with col1:
                st.info(
                    f"👁 Eye State\n\n**{eye_pred}**"
                )

            with col2:
                st.info(
                    f"👄 Mouth State\n\n**{mouth_pred}**"
                )

            fatigue = get_fatigue_level(
                eye_pred,
                mouth_pred
            )

            # =====================================================
            # STORE FATIGUE HISTORY
            # =====================================================

            score = fatigue_score(fatigue)

            st.session_state.fatigue_history.append(score)

            # Keep only latest 20 records
            if len(st.session_state.fatigue_history) > 20:
                st.session_state.fatigue_history.pop(0)

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

            else:

                st.error(
                    "🔴 Driver Status: SEVERE FATIGUE"
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