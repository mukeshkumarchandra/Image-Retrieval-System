import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import urllib.request

# Load pre-trained model
model = models.resnet18(pretrained=True)
model.eval()

# Load ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
with urllib.request.urlopen(LABELS_URL) as f:
    imagenet_classes = [line.strip().decode("utf-8") for line in f.readlines()]

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Streamlit UI
st.set_page_config(page_title="Smart Image Classifier", layout="wide")
st.title("📸 Smart Image Classifier")
st.markdown("Upload **up to 5 images**, and the app will detect the content using a pre-trained **ResNet-18** model. You can also **search** by keyword!")

# Sidebar Search
st.sidebar.header("🔍 Filter Results")
search_term = st.sidebar.text_input("Type a keyword (e.g., cat, laptop, yoga)")

# File uploader
uploaded_files = st.file_uploader("Upload image files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

def classify_image(image):
    img = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
    _, predicted = outputs.max(1)
    return imagenet_classes[predicted.item()]

# Display uploaded images
if uploaded_files:
    matched = 0
    st.markdown("---")
    st.subheader("🔎 Classification Results")

    cols = st.columns(3)

    with st.spinner("Analyzing images..."):
        for idx, file in enumerate(uploaded_files):
            image = Image.open(file).convert("RGB")
            label = classify_image(image)

            if search_term.strip() == "" or search_term.lower() in label.lower():
                with cols[idx % 3]:
                    st.image(image, caption=f"**Prediction:** {label}", use_container_width=True)
                    matched += 1

    if matched == 0:
        st.warning("⚠️ No matching images found for your search term.")
    else:
        st.success(f"✅ Found {matched} matching image(s).")
else:
    st.info("📤 Upload one or more images to begin.")
