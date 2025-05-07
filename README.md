
# 🌊 Underwater Image Enhancement using Multi-Fusion Technique

This project implements an **Underwater Image Enhancement System** using a **Multi-Fusion technique**, built with **Flask** for the web interface and powered by a pre-trained model for enhancing submerged visuals. The system allows users to upload underwater images and view the enhanced outputs instantly.

## Authors
Mahitha S, Srinisha S, Gayathri J

## 📸 Features

- Upload underwater images via a simple web interface
- Enhance images using a multi-fusion-based approach
- View and download the enhanced results
- Lightweight Flask-based server

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/underwater-image-enhancement.git
cd underwater-image-enhancement
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask App

```bash
streamlit run app.py
```

Visit `http://127.0.0.1:5000` in your browser to use the app.

## 🧠 Methodology

The enhancement is done using a **Multi-Fusion technique**, which combines several processed versions of the input image (e.g., white balancing, contrast stretching, and sharpness enhancement) into a single visually pleasing output. The logic is encapsulated in a serialized model (`model.pkl`), loaded and used within the Flask route logic.

## 📁 Input/Output

- **Input**: JPEG/PNG underwater images
- **Output**: Enhanced image viewable and downloadable from the browser

## 🛠 Tech Stack

- Python
- Flask
- OpenCV, NumPy
- HTML/CSS (Jinja templates)

