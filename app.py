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

# ... other imports ...
# ────────────────────────────────────────────────
#  App Setup
# ────────────────────────────────────────────────
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

# ────────────────────────────────────────────────
#  Models
# ────────────────────────────────────────────────
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

# ────────────────────────────────────────────────
#  YOLO Model
# ────────────────────────────────────────────────
model = YOLO("yolov8l.pt")

SHARP_OBJECTS = ["knife", "scissors"]

# ────────────────────────────────────────────────
#  Detection Logic
# ────────────────────────────────────────────────
def detect_and_count(image):
    results = model(image, imgsz=640, conf=0.45)
    annotated = results[0].plot()

    boxes = results[0].boxes
    class_ids = [int(c) for c in boxes.cls]
    class_names = [model.names[i] for i in class_ids]
    counts = dict(Counter(class_names))

    sharp_detected = any(obj in SHARP_OBJECTS for obj in class_names)
    high_risk = False

    person_boxes = []
    sharp_boxes = []

    for box, cls_id in zip(boxes.xyxy, class_ids):
        name = model.names[cls_id]
        if name == "person":
            person_boxes.append(box.cpu().numpy())
        if name in SHARP_OBJECTS:
            sharp_boxes.append(box.cpu().numpy())

    for p in person_boxes:
        px1, py1, px2, py2 = p
        for s in sharp_boxes:
            sx1, sy1, sx2, sy2 = s
            if sx2 > px1 and sx1 < px2 and sy2 > py1 and sy1 < py2:
                high_risk = True
                cv2.rectangle(annotated, (int(sx1), int(sy1)), (int(sx2), int(sy2)), (0, 0, 255), 5)
                cv2.putText(annotated, "HOLDING!", (int(sx1), int(sy1)-15),
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

from datetime import datetime



# ────────────────────────────────────────────────
#  Routes
# ────────────────────────────────────────────────

@app.route('/')
@app.route('/home')
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
            return render_template('register.html', error="Username and password required")

        if User.query.filter_by(username=username).first():
            return render_template('register.html', error="Username already taken")

        hashed = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid username or password")

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/upload_image', methods=['POST'])
@login_required
def upload_image():
    if 'file' not in request.files:
        return "No file", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    img = cv2.imread(path)
    if img is None:
        return "Invalid image", 400

    annotated, counts, alert, high_risk, alert_fn = detect_and_count(img)

    output_filename = f"output_{int(time.time())}.jpg"
    save_temp_image(annotated, output_filename)

    counts_str = ", ".join(f"{k}:{v}" for k,v in counts.items())
    det = Detection(
        user_id=current_user.id,
        objects_detected=counts_str,
        alert=alert,
        high_risk=high_risk,
        image_filename=alert_fn or output_filename
    )
    db.session.add(det)
    db.session.commit()

    return render_template('result.html',
                           image_url=f"/static/{output_filename}",
                           counts=counts,
                           high_risk=high_risk,
                           alert=alert)

@app.route('/upload_video', methods=['POST'])
@login_required
def upload_video():
    if 'file' not in request.files:
        return "No file", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

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
        ret, buffer = cv2.imencode('.jpg', annotated)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

@app.route('/video_feed/<filename>')
@login_required
def video_feed(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(path):
        return "Video not found", 404
    return Response(generate_video(path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam')
@login_required
def webcam():
    return render_template('webcam.html')

@app.route('/detect_webcam', methods=['POST'])
@login_required
def detect_webcam():
    data = request.json.get('image')
    if not data:
        return jsonify({'error': 'No image'}), 400

    encoded = data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    annotated, counts, alert, high_risk, alert_fn = detect_and_count(img)

    _, buffer = cv2.imencode('.jpg', annotated)
    img_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

    counts_str = ", ".join(f"{k}:{v}" for k,v in counts.items())
    det = Detection(
        user_id=current_user.id,
        objects_detected=counts_str,
        alert=alert,
        high_risk=high_risk,
        image_filename=alert_fn
    )
    db.session.add(det)
    db.session.commit()

    alert_msg = ""
    if high_risk:
        alert_msg = '<div class="alert alert-danger">HIGH RISK: Person Holding Sharp Object!</div>'
    elif alert:
        alert_msg = '<div class="alert alert-warning">Sharp Object Detected!</div>'

    return jsonify({
        'image': img_base64,
        'table': f"""
            <table class="table table-bordered table-striped">
                <thead class="table-dark"><tr><th>Object</th><th>Count</th></tr></thead>
                <tbody>
                    {"".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k,v in counts.items())}
                </tbody>
            </table>
        """,
        'alert_msg': alert_msg
    })

@app.route('/history')
@login_required
def history():
    detections = Detection.query.filter_by(user_id=current_user.id)\
                        .order_by(Detection.timestamp.desc())\
                        .limit(20).all()
    return render_template('history.html', detections=detections)

from flask import flash
@app.route('/delete_detection/<int:detection_id>', methods=['POST'])
def delete_detection(detection_id):
    detection = Detection.query.get_or_404(detection_id)
    
    # Optional: delete the image file from disk
    if detection.image_filename:
        image_path = os.path.join(app.config['ALERT_FOLDER'], detection.image_filename)
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception as e:
                print(f"Failed to delete image: {e}")  # or log it properly
    
    db.session.delete(detection)
    db.session.commit()
    
    flash('Detection deleted successfully.', 'success')  # ← this line needs 'flash' imported
    return redirect(url_for('history'))
# ────────────────────────────────────────────────
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)
