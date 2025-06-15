import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# Load mô hình đã huấn luyện
@st.cache_resource
def load_trained_model():
    model = load_model("model_fruit.h5")
    return model

model = load_trained_model()

# Danh sách tên lớp (bạn cần chỉnh sửa đúng theo số lớp mà mô hình bạn huấn luyện)
class_names = ['Cam', 'Chôm chôm', 'Chuối', 'Dâu tây', 'Dưa hấu', 'Dứa', 'Kiwi', 'Măng cụt', 'Táo', 'Xoài'] 

# Giao diện người dùng
st.title("🍎 Nhận Diện Trái Cây")

uploaded_file = st.file_uploader("Tải ảnh trái cây", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh bạn đã tải lên", use_column_width=True)

    # Tiền xử lý ảnh
    img = image.resize((224, 224))  # Kích thước mặc định cho EfficientNetV2-B0
    img_array = np.array(img)
    img_preprocessed = preprocess_input(img_array)
    img_batch = np.expand_dims(img_preprocessed, axis=0)  # (1, 224, 224, 3)


    # Dự đoán
    predictions = model.predict(img_batch)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.markdown(f"### ✅ Dự đoán: **{predicted_class}** ({confidence*100:.2f}%)") sửa hết đi để t copy
