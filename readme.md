# AI Drone Defense (YOLOv5 + Flask + Docker)

This project is a lightweight AI-powered image inference system that detects potential drone threats using a YOLOv5 model deployed via a Flask API and containerized with Docker. Designed with edge deployment in mind, the system simulates a disconnected environment suitable for aerospace and defense applications.

---

## ✨ Features

- ✅ Real-time image detection via REST API (`/detect`)
- ✅ YOLOv5 inference inside a Flask server
- ✅ Dockerized for fast deployment

---

## 🛠 Tech Stack

- Python 3.10
- YOLOv5 (Ultralytics)
- Flask
- Torch + torchvision
- OpenCV
- Docker

---

## 🚀 Quick Start (with Docker)

### 1. Clone and build:

```bash
git clone https://github.com/KianFatemi/AI-Drone-Defense.git
cd AI-Drone-Defense
docker build -t ai-drone-defense .
```

### 2. Run the Server:
```bash
docker run -p 5000:5000 ai-drone-defense
```

## 📸 Example Usage
```bash 
curl -F "file=@your_image.jpg" http://localhost:5000/detect
```

### Sample response:
```json
{
  "detections": [
    {
      "label": "airplane",
      "confidence": 0.8462
    }
  ]
}
```