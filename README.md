# Image-Retrieval-System
Upload up to 5 images, and the app will detect the content using a pre-trained ResNet-18 model. You can also search by keyword!

## 📸 Smart Image Classifier (Streamlit App)

An interactive web app that uses a pre-trained **ResNet-18** model to classify the content of uploaded images. You can upload up to 5 images at once, and the app will display predictions with labels from the **ImageNet** dataset. A built-in **search filter** allows you to find specific content (e.g., "dog", "book", "laptop") across your uploaded images.

### 🔍 Features

* Upload multiple images (JPEG/PNG).
* Automatically classifies image content using deep learning.
* Responsive layout with organized grid display.
* Optional search filter for specific objects or scenes.
* Powered by PyTorch, TorchVision, and Streamlit.

### 🚀 Demo Screenshot

![screenshot](/image%20retriver%20/Image-Retrieval-System/images/tractor.png)

### 🧠 Model Info

* Architecture: `ResNet-18`
* Trained on: `ImageNet (1,000 classes)`
* Framework: PyTorch + TorchVision

### ✅ How to Run

pip install -r requirements.txt
streamlit run app.py