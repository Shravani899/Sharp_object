import os
import time
import base64
import numpy as np
import cv2
from collections import Counter
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = "super-secret-key-change-this-in-production"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.config['ALERT_FOLDER'] = os.path.join('static', 'alerts')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users_and_detections.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
os.makedirs(app.config['ALERT_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    objects_detected = db.Column(db.String(500))
    alert = db.Column(db.Boolean, default=False)
    high_risk = db.Column(db.Boolean, default=False)
    image_filename = db.Column(db.String(200))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load YOLO model
model = YOLO("yolov8m.pt")
print("YOLO Classes:", model.names)

SHARP_OBJECTS = ["knife", "scissors"]

def compute_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def box_center(box):
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    return np.array([x_center, y_center])

def distance_between_boxes(box1, box2):
    c1 = box_center(box1)
    c2 = box_center(box2)
    return np.linalg.norm(c1 - c2)

def detect_and_count(image):
    # Lower conf threshold to 0.1 to catch faint knives
    results = model(image, imgsz=960, conf=0.1, iou=0.35)

    annotated = results[0].plot()
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        print("No detections.")
        return annotated, {}, False, False, None

    class_ids = [int(c) for c in boxes.cls]
    confidences = [float(c) for c in boxes.conf]
    class_names = [model.names[i] for i in class_ids]

    # Debug print all detections with confidence
    print("Detections with confidences:")
    for name, conf in zip(class_names, confidences):
        print(f" - {name}: {conf:.2f}")

    counts = dict(Counter(class_names))

    sharp_detected = any(obj in SHARP_OBJECTS for obj in class_names)
    high_risk = False

    person_boxes = []
    sharp_boxes = []

    for box, cls_id in zip(boxes.xyxy, class_ids):
        name = model.names[cls_id]
        coords = box.cpu().numpy()

        if name == "person":
            person_boxes.append(coords)

        if name in SHARP_OBJECTS:
            sharp_boxes.append(coords)

    # Overlap and proximity logic for high risk
    for p in person_boxes:
        for s in sharp_boxes:
            iou = compute_iou(p, s)
            dist = distance_between_boxes(p, s)

            # Log IoU and distance
            print(f"IoU between person and {model.names[cls_id]}: {iou:.3f}, Distance: {dist:.1f}")

            # Conditions for high risk:
            # Either overlap or very close proximity (e.g., distance < 100 pixels)
            if iou > 0.05 or dist < 100:
                high_risk = True

                sx1, sy1, sx2, sy2 = map(int, s)
                cv2.rectangle(annotated, (sx1, sy1), (sx2, sy2), (0, 0, 255), 5)
                cv2.putText(annotated, "HIGH RISK!", (sx1, sy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    alert = sharp_detected or high_risk

    alert_filename = None
    if alert and current_user.is_authenticated:
        ts = time.strftime("%Y%m%d_%H%M%S")
        alert_filename = f"alert_{current_user.id}_{ts}.jpg"
        alert_path = os.path.join(app.config['ALERT_FOLDER'], alert_filename)
        cv2.imwrite(alert_path, annotated)

    return annotated, counts, alert, high_risk, alert_filename

def save_temp_image(image, filename):
    path = os.path.join(app.config['STATIC_FOLDER'], filename)
    cv2.imwrite(path, image)
    return filename

# You can keep your Flask routes here unchanged...

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)