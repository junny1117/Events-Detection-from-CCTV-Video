import sys
import os
import cv2
import torch
import datetime
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for, session
from flask_socketio import SocketIO, emit
from pathlib import Path
from python.video_stream import VideoStream
from python.detection import ObjectDetector
from python.database import init_db, db_session, Event
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox
from functools import wraps
from threading import Thread

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['TEMPLATES_AUTO_RELOAD'] = True
socketio = SocketIO(app)

USERNAME = 'admin'
PASSWORD = '1234'

# Video stream
video_stream = VideoStream("test.mp4")

weights_path = 'best.pt'
roi_intrusion = (100, 200, 300, 400)
roi_no_parking = (400, 500, 300, 400)
loitering_time_threshold = 30
detector = ObjectDetector(weights_path, roi_intrusion, roi_no_parking, loitering_time_threshold)

tracked_objects = {}
tracked_events = {}

@app.before_request
def setup():
    init_db()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == USERNAME and password == PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='사용자 이름 또는 비밀번호 불일치')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    global roi_intrusion, roi_no_parking
    if request.method == 'POST':
        roi_intrusion = (
            int(request.form['intrusion_x']),
            int(request.form['intrusion_y']),
            int(request.form['intrusion_width']),
            int(request.form['intrusion_height'])
        )
        roi_no_parking = (
            int(request.form['no_parking_x']),
            int(request.form['no_parking_y']),
            int(request.form['no_parking_width']),
            int(request.form['no_parking_height'])
        )
        detector.roi_intrusion = roi_intrusion  
        detector.roi_no_parking = roi_no_parking  
        return redirect(url_for('index'))
    return render_template('settings.html')

def select_roi_async(frame, scale_percent, roi_callback):
    def _select_roi():
        resized_frame = cv2.resize(
            frame,
            (int(frame.shape[1] * scale_percent / 100), int(frame.shape[0] * scale_percent / 100)),
            interpolation=cv2.INTER_AREA
        )

        window_name = "Select ROI"
        cv2.namedWindow(window_name)
        
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) != -1:
            # If window already exists, destroy it to prevent multiple windows
            cv2.destroyWindow(window_name)

        cv2.imshow(window_name, resized_frame)
        roi_resized = cv2.selectROI(window_name, resized_frame, fromCenter=False, showCrosshair=True)

        if roi_resized[2] > 0 and roi_resized[3] > 0:  # Valid ROI check
            roi_original = (
                int(roi_resized[0] / scale_percent * 100),
                int(roi_resized[1] / scale_percent * 100),
                int(roi_resized[2] / scale_percent * 100),
                int(roi_resized[3] / scale_percent * 100)
            )
            roi_callback(roi_original)
        else:
            print("No valid ROI selected or selection was canceled.")

        cv2.destroyWindow(window_name)

    # Run the ROI selection in a separate thread
    thread = Thread(target=_select_roi)
    thread.start()
    thread.join()  # Wait for the thread to finish

@app.route('/select_roi')
@login_required
def select_roi():
    frame = video_stream.get_frame()
    if frame is not None:
        select_roi_async(frame, 50, lambda roi: update_roi('roi_intrusion', roi))
    return redirect(url_for('index'))

@app.route('/select_roi2')
@login_required
def select_roi2():
    frame = video_stream.get_frame()
    if frame is not None:
        select_roi_async(frame, 50, lambda roi: update_roi('roi_no_parking', roi))
    return redirect(url_for('index'))

def update_roi(roi_type, roi):
    if roi_type == 'roi_intrusion':
        detector.roi_intrusion = roi
    elif roi_type == 'roi_no_parking':
        detector.roi_no_parking = roi
    print(f"{roi_type} updated to {roi}")

@app.route('/events')
@login_required
def events():
    events = db_session.query(Event).order_by(Event.timestamp.desc()).all()  # 최신 이벤트부터 정렬
    return render_template('events.html', events=events)

def gen_frames():
    global tracked_objects, tracked_events
    while True:
        frame = video_stream.get_frame()
        if frame is None:
            break

        frame, detected_events = detector.detect_and_draw(frame)

        current_time = datetime.datetime.now()
        for event in detected_events:
            event_id = f"{event['type']}_{int(event['신뢰도'] * 100)}"  # Create a unique ID for each event

            if event_id not in tracked_events or (current_time - tracked_events[event_id]).total_seconds() > 10:
                print(f"New event detected: {event_id}")  
                new_event = Event(
                    label=event['type'],
                    confidence=event['신뢰도'],
                    timestamp=current_time
                )
                db_session.add(new_event)
                db_session.commit()
                socketio.emit('new_event', {
                    'label': new_event.label,
                    'confidence': new_event.confidence,
                    'timestamp': new_event.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                })
                tracked_events[event_id] = new_event.timestamp  

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
