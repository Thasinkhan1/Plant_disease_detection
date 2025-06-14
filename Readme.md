# 🌿 Plant Disease Detection App

> **"Early detection. Better protection."**  
> An AI-powered tool to identify plant diseases, using transfer learning by customizing with different techniques like initialization, regularization, dropout, GlobalAveragePooling, to improve the model performance and reduce overfitting.  
> Built with Docker containerization, and currently working on integrating a chatbot into this project that answers your queries live related to plant disease prevention and care.

---

## 🔥 Live Demo

🎥 **Watch the app in action:**  
📌 To watch the video, please open the attached `.mp4` file.

---

## 🌱 Project Highlights

✅ 38 plant disease classes  
✅ 70,000+ images used for training  
✅ CNN with BatchNorm & Dropout for accuracy  
✅ Disease prevention tips included

---

## 🧠 How It Works

1. 📷 **Upload** a leaf image  
2. 🤖 **Model** detects the disease  
3. 📌 **Displays** disease name with confidence percentage  

## 🛠️ Tech Stack

| Category     | Tools Used                                  |
|--------------|---------------------------------------------|
| Model        | Python, TensorFlow/Keras                    |
| UI/UX        | Streamlit                                   |
| Data         | PlantVillage Dataset (via Kaggle)           |
| Others       | Pandas, NumPy, OpenCV, Matplotlib, scikit-learn |

---

## 🚀 Quick Start (Local Setup)

```bash
# Clone the repo
git clone https://github.com/Thasinkhan1/Plant_disease_detection

# Move into the directory
cd plant-disease-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py  
