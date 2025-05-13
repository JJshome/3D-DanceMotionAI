"""
Web interface for DanceMotionAI system.
This module implements a Flask-based web interface for the DanceMotionAI system, 
enabling users to upload videos, analyze dance movements, and visualize results through a browser.
"""

import os
import sys
import tempfile
import uuid
import json
import time
import threading
import queue
import numpy as np
import torch
import yaml
from typing import Dict, List, Tuple, Optional, Union, Any
from flask import Flask, request, render_template, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename

# Add the root directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import DanceMotionAI modules
from models.dancehrnet import create_model as create_dancehrnet
from models.hands_4d import Hands4D
from models.danceformer import create_model as create_danceformer
from models.dance_dtw import create_model as create_dance_dtw
from models.music_analysis import create_music_analyzer
from utils.visualization import create_pose_animation

# Import necessary functions from integrated_demo
from examples.integrated_demo import (
    load_config, 
    load_video, 
    extract_audio, 
    preprocess_frames, 
    process_video, 
    analyze_music,
    correlate_dance_and_music,
    compare_choreographies,
    compute_dance_metrics
)

app = Flask(__name__, static_folder='static', template_folder='templates')

# Configure uploads
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

# Create required directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(os.path.join(RESULT_FOLDER, 'visualizations'), exist_ok=True)
os.makedirs(os.path.join(RESULT_FOLDER, 'animations'), exist_ok=True)

# Global variables
config = None
models = None
device = None
processing_queue = queue.Queue()
processing_results = {}
processing_status = {}

def allowed_file(filename):
    """Check if a file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models_for_server():
    """Load models for the server."""
    global config, models, device
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    config_path = os.path.join('configs', 'default_config.yaml')
    config = load_config(config_path)
    
    # Create model instances
    dancehrnet = create_dancehrnet(config['pose_estimation'])
    hands_4d = Hands4D(config['hands_4d'])
    danceformer = create_danceformer(config['multimodal_analysis'])
    dance_dtw = create_dance_dtw(config['similarity_analysis'])
    music_analyzer = create_music_analyzer(config['music_analysis'])
    
    # Move models to device
    dancehrnet = dancehrnet.to(device)
    hands_4d = hands_4d.to(device)
    danceformer = danceformer.to(device)
    dance_dtw = dance_dtw.to(device)
    
    # Set models to evaluation mode
    dancehrnet.eval()
    hands_4d.eval()
    danceformer.eval()
    dance_dtw.eval()
    
    # Store models in a dictionary
    models = {
        'dancehrnet': dancehrnet,
        'hands_4d': hands_4d,
        'danceformer': danceformer,
        'dance_dtw': dance_dtw,
        'music_analyzer': music_analyzer
    }
    
    print("All models loaded successfully")

def process_worker():
    """Background worker to process videos in the queue."""
    while True:
        try:
            task_id, video_path, reference_path, task_type = processing_queue.get()
            
            # Update status
            processing_status[task_id] = {
                'status': 'processing',
                'progress': 0,
                'message': 'Starting processing'
            }
            
            # Create result directory for this task
            task_dir = os.path.join(app.config['RESULT_FOLDER'], task_id)
            os.makedirs(task_dir, exist_ok=True)
            
            try:
                # Load video
                update_status(task_id, 5, 'Loading video')
                frames, fps = load_video(video_path)
                
                # Extract audio
                update_status(task_id, 10, 'Extracting audio')
                audio_path = os.path.join(task_dir, 'audio.wav')
                extract_audio(video_path, audio_path)
                
                # Preprocess frames
                update_status(task_id, 15, 'Preprocessing frames')
                processed_frames = preprocess_frames(frames)
                
                # Process video (3D pose estimation)
                update_status(task_id, 20, 'Estimating 3D poses')
                poses_3d, hand_poses = process_video(
                    processed_frames, 
                    models['dancehrnet'], 
                    models['hands_4d'],
                    device
                )
                
                # Analyze music
                update_status(task_id, 40, 'Analyzing music')
                music_features = analyze_music(audio_path, models['music_analyzer'])
                
                # Correlate dance and music
                update_status(task_id, 60, 'Correlating dance and music')
                multimodal_features = correlate_dance_and_music(
                    poses_3d, 
                    music_features, 
                    models['danceformer'],
                    device
                )
                
                # If reference video is provided, compare choreographies
                similarity_results = None
                if reference_path and task_type == 'compare':
                    update_status(task_id, 70, 'Processing reference video')
                    ref_frames, ref_fps = load_video(reference_path)
                    ref_processed_frames = preprocess_frames(ref_frames)
                    
                    # Process reference video
                    update_status(task_id, 75, 'Estimating reference 3D poses')
                    ref_poses_3d, ref_hand_poses = process_video(
                        ref_processed_frames, 
                        models['dancehrnet'], 
                        models['hands_4d'],
                        device
                    )
                    
                    # Extract audio from reference
                    update_status(task_id, 80, 'Analyzing reference audio')
                    ref_audio_path = os.path.join(task_dir, 'ref_audio.wav')
                    extract_audio(reference_path, ref_audio_path)
                    ref_music_features = analyze_music(ref_audio_path, models['music_analyzer'])
                    
                    # Correlate reference dance and music
                    update_status(task_id, 85, 'Correlating reference dance and music')
                    ref_multimodal_features = correlate_dance_and_music(
                        ref_poses_3d, 
                        ref_music_features, 
                        models['danceformer'],
                        device
                    )
                    
                    # Compare choreographies
                    update_status(task_id, 90, 'Comparing choreographies')
                    similarity_results = compare_choreographies(
                        multimodal_features, 
                        ref_multimodal_features, 
                        models['dance_dtw']
                    )
                
                # Compute dance metrics
                update_status(task_id, 95, 'Computing dance metrics')
                metrics = compute_dance_metrics(poses_3d, hand_poses)
                
                # Create visualizations
                update_status(task_id, 98, 'Creating visualizations')
                visualization_path = os.path.join(task_dir, 'visualization.mp4')
                create_pose_animation(
                    frames, 
                    poses_3d, 
                    visualization_path, 
                    fps=fps
                )
                
                # Save results
                results = {
                    'metrics': metrics,
                    'visualization_path': visualization_path,
                    'task_type': task_type
                }
                
                if similarity_results:
                    results['similarity'] = similarity_results
                
                # Save results to JSON file
                results_path = os.path.join(task_dir, 'results.json')
                with open(results_path, 'w') as f:
                    json.dump(results, f)
                
                processing_results[task_id] = results
                update_status(task_id, 100, 'Processing complete')
                
            except Exception as e:
                print(f"Error processing task {task_id}: {str(e)}")
                processing_status[task_id] = {
                    'status': 'error',
                    'progress': 0,
                    'message': f'Error: {str(e)}'
                }
            
            finally:
                processing_queue.task_done()
                
        except Exception as e:
            print(f"Worker error: {str(e)}")
            time.sleep(1)  # Prevent CPU spinning on repeated errors

def update_status(task_id, progress, message):
    """Update the status of a processing task."""
    processing_status[task_id] = {
        'status': 'processing',
        'progress': progress,
        'message': message
    }
    print(f"Task {task_id}: {progress}% - {message}")

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
    file.save(file_path)
    
    # Determine task type
    task_type = request.form.get('task_type', 'analyze')
    
    # If this is a comparison task, check for reference file
    reference_path = None
    if task_type == 'compare' and 'reference_file' in request.files:
        ref_file = request.files['reference_file']
        if ref_file.filename != '' and allowed_file(ref_file.filename):
            ref_filename = secure_filename(ref_file.filename)
            reference_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_ref_{ref_filename}")
            ref_file.save(reference_path)
    
    # Initialize status
    processing_status[task_id] = {
        'status': 'queued',
        'progress': 0,
        'message': 'Queued for processing'
    }
    
    # Add task to processing queue
    processing_queue.put((task_id, file_path, reference_path, task_type))
    
    return jsonify({
        'task_id': task_id,
        'status': 'queued',
        'message': 'File uploaded successfully, processing started'
    })

@app.route('/status/<task_id>')
def get_status(task_id):
    """Get the status of a processing task."""
    if task_id not in processing_status:
        return jsonify({'error': 'Task not found'}), 404
    
    return jsonify(processing_status[task_id])

@app.route('/results/<task_id>')
def get_results(task_id):
    """Get the results of a processing task."""
    if task_id not in processing_results:
        # Check if results file exists
        results_path = os.path.join(app.config['RESULT_FOLDER'], task_id, 'results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                processing_results[task_id] = json.load(f)
        else:
            return jsonify({'error': 'Results not found'}), 404
    
    return jsonify(processing_results[task_id])

@app.route('/visualization/<task_id>')
def get_visualization(task_id):
    """Stream visualization video for a processing task."""
    task_dir = os.path.join(app.config['RESULT_FOLDER'], task_id)
    return send_from_directory(task_dir, 'visualization.mp4')

@app.route('/view/<task_id>')
def view_results(task_id):
    """Render the results view page."""
    if task_id not in processing_status:
        return redirect(url_for('index'))
    
    status = processing_status[task_id]
    
    if status['status'] != 'complete':
        # Show processing page with progress
        return render_template('processing.html', task_id=task_id, status=status)
    
    # Show results page
    return render_template('results.html', task_id=task_id)

def start_background_workers():
    """Start background worker threads."""
    # Start worker thread
    worker_thread = threading.Thread(target=process_worker, daemon=True)
    worker_thread.start()

if __name__ == '__main__':
    # Load models
    load_models_for_server()
    
    # Start background workers
    start_background_workers()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
