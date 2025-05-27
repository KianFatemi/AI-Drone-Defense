FROM python:3.10-slim

WORKDIR /app

# Install system packages for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --upgrade pip
RUN pip install flask torch torchvision pillow opencv-python

# Install YOLOv5 dependencies
WORKDIR /app/yolov5
RUN pip install -r requirements.txt

WORKDIR /app

CMD ["python", "app.py"]