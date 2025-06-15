import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load mô hình
model = load_model("model_fruit.h5")

# Danh sách lớp
classes = ['Cam', 'Chôm chôm', 'Chuối', 'Dâu tây', 'Dứa', 'Dưa hấu', 'Kiwi', 'Măng cụt', 'Táo', 'Xoài']

st.title("🍓 Nhận diện trái cây")

file = st.file_uploader("Tải ảnh trái cây lên", type=["jpg", "png", "jpeg"])
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Ảnh đã chọn", use_column_width=True)

    # Tiền xử lý
    img = img.resize((224, 224))
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # Dự đoán
    pred = model.predict(img)
    label = classes[np.argmax(pred)]
    score = np.max(pred)

    st.success(f"Kết quả: **{label}** – độ tin cậy: {score:.2%}")
