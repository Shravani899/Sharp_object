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

# ───────────────────────────────
# App Setup
# ───────────────────────────────
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

# ───────────────────────────────
# Models
# ───────────────────────────────
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

# ───────────────────────────────
# YOLOv8 Model Setup
# ───────────────────────────────
model = YOLO("yolov8m.pt")  # COCO pretrained
print("YOLO Classes:", model.names)

# Objects considered sharp or risky
SHARP_OBJECTS = ["knife", "scissors", "fork"]
CONTEXT_OBJECTS = ["dining table", "bowl"]

# ───────────────────────────────
# Helper: Compute IoU for overlap detection
# ───────────────────────────────
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / (boxAArea + boxBArea - interArea + 1e-6)

# ───────────────────────────────
# Detection Logic
# ───────────────────────────────
def detect_and_count(image):
    results = model(image, imgsz=1280, conf=0.1, iou=0.3)
    annotated = results[0].plot()
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return annotated, {}, False, False, None

    class_ids = [int(c) for c in boxes.cls]
    class_names = [model.names[i] for i in class_ids]
    counts = dict(Counter(class_names))

    # Detect sharp objects
    sharp_detected = any(obj in SHARP_OBJECTS for obj in class_names)

    # Fallback: person + context → assume risk
    if not sharp_detected:
        if "person" in class_names and any(obj in class_names for obj in CONTEXT_OBJECTS):
            sharp_detected = True

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

    # IoU-based overlap
    for p in person_boxes:
        for s in sharp_boxes:
            if compute_iou(p, s) > 0.05:
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

# ───────────────────────────────
# Routes
# ───────────────────────────────
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            return render_template('register.html', error="Required fields missing")
        if User.query.filter_by(username=username).first():
            return render_template('register.html', error="Username exists")
        user = User(username=username, password=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and check_password_hash(user.password, request.form.get('password')):
            login_user(user)
            return redirect(url_for('dashboard'))
        return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ───────── IMAGE ─────────
@app.route('/upload_image', methods=['POST'])
@login_required
def upload_image():
    file = request.files.get('file')
    if not file or file.filename == '':
        return "No file", 400
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)
    img = cv2.imread(path)
    annotated, counts, alert, high_risk, alert_fn = detect_and_count(img)
    output_filename = f"output_{int(time.time())}.jpg"
    save_temp_image(annotated, output_filename)
    db.session.add(Detection(
        user_id=current_user.id,
        objects_detected=str(counts),
        alert=alert,
        high_risk=high_risk,
        image_filename=alert_fn or output_filename
    ))
    db.session.commit()
    return render_template('result.html',
                           image_url=f"/static/{output_filename}",
                           counts=counts,
                           alert=alert,
                           high_risk=high_risk)

# ───────── VIDEO ─────────
@app.route('/upload_video', methods=['POST'])
@login_required
def upload_video():
    file = request.files.get('file')
    if not file or file.filename == '':
        return "No file", 400
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)
    return redirect(url_for('video_feed', filename=filename))

def generate_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        annotated, _, _, _, _ = detect_and_count(frame)
        _, buffer = cv2.imencode('.jpg', annotated)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')
    cap.release()

@app.route('/video_feed/<filename>')
@login_required
def video_feed(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(path):
        return "Video not found", 404
    return Response(generate_video(path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ───────── WEBCAM ─────────
@app.route('/webcam')
@login_required
def webcam():
    return render_template('webcam.html')

@app.route('/detect_webcam', methods=['POST'])
@login_required
def detect_webcam():
    data = request.json.get('image', None)
    encoded = data.split(',')[1]
    img = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
    annotated, counts, alert, high_risk, alert_fn = detect_and_count(img)
    _, buffer = cv2.imencode('.jpg', annotated)
    img_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()
    db.session.add(Detection(
        user_id=current_user.id,
        objects_detected=str(counts),
        alert=alert,
        high_risk=high_risk,
        image_filename=alert_fn
    ))
    db.session.commit()
    return jsonify({
        'image': img_base64,
        'counts': counts,
        'alert': alert,
        'high_risk': high_risk
    })

# ───────── HISTORY ─────────
@app.route('/history')
@login_required
def history():
    detections = Detection.query.filter_by(user_id=current_user.id)\
        .order_by(Detection.timestamp.desc()).limit(20).all()
    return render_template('history.html', detections=detections)

@app.route('/delete_detection/<int:detection_id>', methods=['POST'])
@login_required
def delete_detection(detection_id):
    detection = Detection.query.get_or_404(detection_id)
    if detection.image_filename:
        path = os.path.join(app.config['ALERT_FOLDER'], detection.image_filename)
        if os.path.exists(path):
            os.remove(path)
    db.session.delete(detection)
    db.session.commit()
    flash('Deleted successfully', 'success')
    return redirect(url_for('history'))

# ───────────────────────────────
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)

