# realtime_bag_counter_pc.py
# PC optimized version for desktop/laptop computers

import cv2
import numpy as np
from ultralytics import YOLO
import os
import threading
import time
from datetime import datetime
from collections import defaultdict
import signal
import sys
import torch
import logging
from pathlib import Path
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# For Excel - using xlsxwriter as it's more lightweight than openpyxl
try:
    import xlsxwriter
    EXCEL_AVAILABLE = True
except ImportError:
    logger.warning("xlsxwriter not found. Installing...")
    os.system("pip install xlsxwriter")
    import xlsxwriter
    EXCEL_AVAILABLE = True

# ===================== PC OPTIMIZATIONS =====================
# Enable multi-threading for better PC performance
os.environ['OMP_NUM_THREADS'] = str(min(8, psutil.cpu_count()))
os.environ['MKL_NUM_THREADS'] = str(min(8, psutil.cpu_count()))

# ===================== CONFIGURATION =====================
class Config:
    # File paths - CHANGE THESE FOR YOUR SETUP
    MODEL_PATH = r"D:\ready\bag_counter\yolo_model\best.pt"
    CAMERA_INDEX = 0
    
    # Video settings optimized for PC
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FPS = 30
    
    # Detection settings
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    
    # Tracking settings
    TRACK_DISTANCE_THRESHOLD = 150
    TRACK_MIN_HITS = 3
    TRACK_MAX_AGE = 30
    
    # Performance settings - more aggressive for PC
    SKIP_FRAMES = 1  # Process every nth frame (1 = process all frames)
    MAX_DETECTIONS = 100  # Higher limit for PC
    BATCH_SIZE = 1
    
    # Camera retry settings
    CAMERA_RETRY_COUNT = 3
    CAMERA_RETRY_DELAY = 1
    CAMERA_WARMUP_FRAMES = 5
    
    # Save settings
    SAVE_INTERVAL = 60
    OUTPUT_DIR = r"D:\ready\bag_counter\bag_counts"
    
    # Display settings
    DISPLAY_SCALE = 0.8  # Scale factor for display window
    SHOW_CLASS_NAMES = True
    SHOW_CONFIDENCE = True

# ===================== CLASS NAMES =====================
class_names = [
    "14% روا د بیاض دواجن",
    "14% روا د تسمین مواشی", 
    "16% روا د حلا ب مواشی",
    "16% روا د بیاض دواجن",
    "16% روا د تسمین مواشی",
    "19% روا د حلا ب عالی الإدار مواشی",
    "19% روا د سوبر دواجن",
    "20% روا د فطام بتلو مواشی",
    "21% روا د سوبر دواجن",
    "21% روا د بادی نامی محبوب دواجن",
    "21% روا د بادی نامی مفتت دواجن",
    "23% روا د سوبر دواجن"
]

# ===================== PC CAMERA UTILITIES =====================
class PCCameraUtils:
    @staticmethod
    def get_available_cameras():
        """Get list of available cameras on PC"""
        available_cameras = []
        
        # Check DirectShow cameras (Windows)
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
            else:
                # Try other backends
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available_cameras.append(i)
                    cap.release()
        
        return available_cameras
    
    @staticmethod
    def get_camera_capabilities(index):
        """Get camera capabilities and supported resolutions"""
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
        
        if not cap.isOpened():
            return None
        
        # Common resolutions to test
        test_resolutions = [
            (1920, 1080),  # 1080p
            (1280, 720),   # 720p
            (960, 720),    # 720p alternative
            (800, 600),    # SVGA
            (640, 480),    # VGA
        ]
        
        supported_resolutions = []
        for width, height in test_resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if actual_width == width and actual_height == height:
                supported_resolutions.append((width, height))
        
        capabilities = {
            'supported_resolutions': supported_resolutions,
            'current_width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'current_height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'backend': cap.getBackendName()
        }
        
        cap.release()
        return capabilities

# ===================== ENHANCED TRACKER FOR PC =====================
class EnhancedTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.unique_bags = {name: set() for name in class_names}
        self.track_to_class = {}
        self.track_to_confidence = {}
        self.frame_count = 0
        self.confidence_history = defaultdict(list)
        
    def calculate_distance(self, box1, box2):
        """Calculate center-to-center distance"""
        center1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
        center2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def update(self, detections):
        """Enhanced tracking with confidence smoothing"""
        self.frame_count += 1
        
        # Age existing tracks
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['age'] += 1
            if self.tracks[track_id]['age'] > Config.TRACK_MAX_AGE:
                if track_id in self.track_to_class:
                    del self.track_to_class[track_id]
                if track_id in self.track_to_confidence:
                    del self.track_to_confidence[track_id]
                if track_id in self.confidence_history:
                    del self.confidence_history[track_id]
                del self.tracks[track_id]
        
        if len(detections) == 0:
            return self.get_confirmed_tracks()
        
        # Limit detections for performance
        if len(detections) > Config.MAX_DETECTIONS:
            detections = sorted(detections, key=lambda x: x[4], reverse=True)[:Config.MAX_DETECTIONS]
        
        # Hungarian algorithm-style matching
        cost_matrix = []
        track_ids = [tid for tid, track in self.tracks.items() if track['age'] <= Config.TRACK_MAX_AGE]
        
        for track_id in track_ids:
            track_box = self.tracks[track_id]['bbox']
            costs = []
            
            for det in detections:
                det_box = det[:4]
                distance = self.calculate_distance(track_box, det_box)
                iou = self.calculate_iou(track_box, det_box)
                
                if distance < Config.TRACK_DISTANCE_THRESHOLD or iou > 0.1:
                    cost = distance * (1 - iou)  # Lower cost is better
                else:
                    cost = float('inf')
                
                costs.append(cost)
            
            cost_matrix.append(costs)
        
        # Simple greedy matching (for better performance, could use Hungarian algorithm)
        matched_detections = set()
        matched_tracks = set()
        
        while True:
            min_cost = float('inf')
            best_track_idx = -1
            best_det_idx = -1
            
            for track_idx, costs in enumerate(cost_matrix):
                if track_idx in matched_tracks:
                    continue
                for det_idx, cost in enumerate(costs):
                    if det_idx in matched_detections or cost == float('inf'):
                        continue
                    if cost < min_cost:
                        min_cost = cost
                        best_track_idx = track_idx
                        best_det_idx = det_idx
            
            if min_cost == float('inf'):
                break
            
            # Match found
            track_id = track_ids[best_track_idx]
            det = detections[best_det_idx]
            x1, y1, x2, y2, conf, cls_id = det
            
            # Update track with confidence smoothing
            self.confidence_history[track_id].append(conf)
            if len(self.confidence_history[track_id]) > 10:
                self.confidence_history[track_id].pop(0)
            
            smoothed_confidence = np.mean(self.confidence_history[track_id])
            
            self.tracks[track_id].update({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': conf,
                'smoothed_confidence': smoothed_confidence,
                'class_id': int(cls_id),
                'age': 0,
                'hits': self.tracks[track_id]['hits'] + 1,
                'last_seen': self.frame_count
            })
            
            self.track_to_confidence[track_id] = smoothed_confidence
            
            matched_detections.add(best_det_idx)
            matched_tracks.add(best_track_idx)
        
        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_detections:
                x1, y1, x2, y2, conf, cls_id = det
                
                self.tracks[self.next_id] = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'smoothed_confidence': conf,
                    'class_id': int(cls_id),
                    'age': 0,
                    'hits': 1,
                    'last_seen': self.frame_count
                }
                
                self.confidence_history[self.next_id] = [conf]
                self.track_to_confidence[self.next_id] = conf
                self.next_id += 1
        
        return self.get_confirmed_tracks()
    
    def get_confirmed_tracks(self):
        """Get confirmed tracks and update unique bags"""
        confirmed_tracks = []
        
        for track_id, track in self.tracks.items():
            if track['hits'] >= Config.TRACK_MIN_HITS and track['age'] == 0:
                confirmed_tracks.append([
                    track['bbox'][0], track['bbox'][1], 
                    track['bbox'][2], track['bbox'][3], 
                    track_id, track['smoothed_confidence']
                ])
                
                # Add to unique bags
                cls_id = track['class_id']
                if cls_id < len(class_names):
                    label = class_names[cls_id]
                    if track_id not in self.track_to_class:
                        self.track_to_class[track_id] = cls_id
                        self.unique_bags[label].add(track_id)
                        logger.info(f"🆕 New bag: ID {track_id} - {label} (conf: {track['smoothed_confidence']:.2f})")
        
        return confirmed_tracks

# ===================== ENHANCED PERFORMANCE MONITOR =====================
class PCPerformanceMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.processing_times = []
        self.fps_times = []
        self.gpu_usage = []
        self.cpu_usage = []
        self.memory_usage = []
        self.last_time = time.time()
        
        # Check if GPU is available
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            logger.info(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("💻 Using CPU for inference")
    
    def update(self, processing_time):
        current_time = time.time()
        
        # Update processing times
        self.processing_times.append(processing_time)
        if len(self.processing_times) > self.window_size:
            self.processing_times.pop(0)
        
        # Update FPS
        frame_time = current_time - self.last_time
        self.fps_times.append(frame_time)
        if len(self.fps_times) > self.window_size:
            self.fps_times.pop(0)
        
        # System monitoring
        self.cpu_usage.append(psutil.cpu_percent())
        if len(self.cpu_usage) > self.window_size:
            self.cpu_usage.pop(0)
        
        memory_info = psutil.virtual_memory()
        self.memory_usage.append(memory_info.percent)
        if len(self.memory_usage) > self.window_size:
            self.memory_usage.pop(0)
        
        # GPU monitoring if available
        if self.gpu_available:
            try:
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                self.gpu_usage.append(gpu_memory)
                if len(self.gpu_usage) > self.window_size:
                    self.gpu_usage.pop(0)
            except:
                pass
        
        self.last_time = current_time
    
    def get_stats(self):
        if not self.processing_times or not self.fps_times:
            return {"avg_processing_ms": 0, "avg_fps": 0}
        
        stats = {
            "avg_processing_ms": np.mean(self.processing_times) * 1000,
            "avg_fps": 1.0 / np.mean(self.fps_times) if np.mean(self.fps_times) > 0 else 0,
            "min_processing_ms": min(self.processing_times) * 1000,
            "max_processing_ms": max(self.processing_times) * 1000,
            "avg_cpu_percent": np.mean(self.cpu_usage) if self.cpu_usage else 0,
            "avg_memory_percent": np.mean(self.memory_usage) if self.memory_usage else 0
        }
        
        if self.gpu_available and self.gpu_usage:
            stats["avg_gpu_memory_gb"] = np.mean(self.gpu_usage)
        
        return stats

# ===================== ENHANCED EXCEL SAVER =====================
class EnhancedExcelSaver:
    def __init__(self):
        self.output_dir = Path(Config.OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_results(self, unique_bags, confidence_data=None, suffix="", performance_stats=None):
        """Enhanced Excel saving with confidence data and better formatting"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bag_count_{timestamp}{suffix}.xlsx"
        filepath = self.output_dir / filename
        
        try:
            workbook = xlsxwriter.Workbook(str(filepath), {'nan_inf_to_errors': True})
            
            # Create formats
            header_format = workbook.add_format({
                'bold': True,
                'font_size': 12,
                'bg_color': '#4F81BD',
                'font_color': 'white',
                'border': 1
            })
            
            total_format = workbook.add_format({
                'bold': True,
                'bg_color': '#FFEB9C',
                'border': 1
            })
            
            # Main results sheet
            worksheet = workbook.add_worksheet('عدد الأكياس')
            worksheet.set_column('A:A', 40)  # Wider for Arabic text
            worksheet.set_column('B:D', 15)
            worksheet.set_column('E:E', 25)
            
            headers = ['النوع', 'العدد', 'متوسط الثقة', 'معرفات الأكياس', 'وقت الحفظ']
            for col, header in enumerate(headers):
                worksheet.write(0, col, header, header_format)
            
            row = 1
            total_count = 0
            
            for name, ids in unique_bags.items():
                count = len(ids)
                if count > 0:
                    ids_str = ", ".join(map(str, sorted(ids)))
                    avg_confidence = 0
                    
                    if confidence_data:
                        confidences = [confidence_data.get(id, 0) for id in ids if id in confidence_data]
                        avg_confidence = np.mean(confidences) if confidences else 0
                    
                    worksheet.write(row, 0, name)
                    worksheet.write(row, 1, count)
                    worksheet.write(row, 2, f"{avg_confidence:.2f}")
                    worksheet.write(row, 3, ids_str)
                    worksheet.write(row, 4, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    total_count += count
                    row += 1
            
            # Total row
            worksheet.write(row, 0, 'المجموع الكلي', total_format)
            worksheet.write(row, 1, total_count, total_format)
            worksheet.write(row, 4, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), total_format)
            
            # Performance stats sheet
            if performance_stats:
                perf_sheet = workbook.add_worksheet('Performance')
                perf_sheet.write(0, 0, 'Metric', header_format)
                perf_sheet.write(0, 1, 'Value', header_format)
                
                row = 1
                for key, value in performance_stats.items():
                    perf_sheet.write(row, 0, key.replace('_', ' ').title())
                    if isinstance(value, float):
                        perf_sheet.write(row, 1, f"{value:.2f}")
                    else:
                        perf_sheet.write(row, 1, str(value))
                    row += 1
            
            workbook.close()
            logger.info(f"💾 Results saved: {filename} (Total: {total_count} bags)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving Excel: {e}")
            return False

# ===================== MAIN PC APPLICATION =====================
class PCRealTimeBagCounter:
    def __init__(self):
        self.running = True
        self.model = None
        self.cap = None
        self.tracker = EnhancedTracker()
        self.excel_saver = EnhancedExcelSaver()
        self.performance_monitor = PCPerformanceMonitor()
        self.last_save_time = time.time()
        self.frame_count = 0
        self.processed_frame_count = 0
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"🔄 Received signal {signum}. Shutting down gracefully...")
        self.running = False
    
    def load_model_optimized(self):
        """Load YOLO model with PC optimizations"""
        logger.info(f"📁 Model path: {Config.MODEL_PATH}")
        
        if not os.path.exists(Config.MODEL_PATH):
            logger.error(f"❌ Model not found: {Config.MODEL_PATH}")
            return False
        
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"🔥 Loading model on {device.upper()}...")
            
            self.model = YOLO(Config.MODEL_PATH)
            
            # Move model to GPU if available
            if torch.cuda.is_available():
                self.model.to('cuda')
                logger.info("🚀 Model loaded on GPU")
            else:
                logger.info("💻 Model loaded on CPU")
            
            # Warmup
            logger.info("🔥 Warming up model...")
            dummy_input = np.zeros((Config.FRAME_HEIGHT, Config.FRAME_WIDTH, 3), dtype=np.uint8)
            self.model(dummy_input, verbose=False)
            
            logger.info("✅ Model loaded and warmed up successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False
    
    def initialize_camera_pc(self):
        """Initialize camera optimized for PC"""
        logger.info("📹 Initializing PC camera...")
        
        # Get available cameras
        available_cameras = PCCameraUtils.get_available_cameras()
        if not available_cameras:
            logger.error("❌ No cameras found")
            return False
        
        logger.info(f"📷 Available cameras: {available_cameras}")
        
        # Try primary camera index first
        camera_indices = [Config.CAMERA_INDEX] + [i for i in available_cameras if i != Config.CAMERA_INDEX]
        
        # Try different backends (prioritize DirectShow for Windows)
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_ANY, "Any")
        ]
        
        for camera_idx in camera_indices:
            logger.info(f"🎯 Trying camera {camera_idx}...")
            
            # Get camera capabilities
            capabilities = PCCameraUtils.get_camera_capabilities(camera_idx)
            if capabilities:
                logger.info(f"   Supported resolutions: {capabilities['supported_resolutions']}")
            
            for backend, backend_name in backends:
                try:
                    logger.info(f"   Testing {backend_name} backend...")
                    
                    if self.cap:
                        self.cap.release()
                        time.sleep(0.5)
                    
                    self.cap = cv2.VideoCapture(camera_idx, backend)
                    
                    if not self.cap.isOpened():
                        logger.warning(f"   Failed to open with {backend_name}")
                        continue
                    
                    # Set optimal properties
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
                    self.cap.set(cv2.CAP_PROP_FPS, Config.FPS)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Enable auto-exposure and auto-focus if available
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                    
                    # Get actual settings
                    actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                    
                    logger.info(f"   📐 Resolution: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
                    
                    # Test frame capture
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        logger.info(f"   ✅ Frame test successful: {test_frame.shape}")
                        
                        # Warmup
                        for _ in range(Config.CAMERA_WARMUP_FRAMES):
                            self.cap.read()
                        
                        logger.info("✅ Camera initialization successful!")
                        return True
                    else:
                        logger.warning(f"   ❌ Frame capture failed")
                        
                except Exception as e:
                    logger.warning(f"   Error with {backend_name}: {e}")
                    continue
        
        logger.error("❌ Failed to initialize camera")
        return False
    
    def process_frame_optimized(self, frame):
        """Optimized frame processing for PC"""
        start_time = time.time()
        
        try:
            # Run YOLO detection
            results = self.model(
                frame, 
                conf=Config.CONFIDENCE_THRESHOLD,
                iou=Config.IOU_THRESHOLD,
                verbose=False,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for box in boxes:
                    try:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        
                        if cls_id < len(class_names) and conf >= Config.CONFIDENCE_THRESHOLD:
                            detections.append([x1, y1, x2, y2, conf, cls_id])
                    except Exception as e:
                        logger.warning(f"Error processing detection: {e}")
                        continue
            
            # Update tracker
            tracks = self.tracker.update(detections)
            
            # Draw results
            self.draw_results_enhanced(frame, tracks)
            
            # Update performance monitoring
            processing_time = time.time() - start_time
            self.performance_monitor.update(processing_time)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame
    
    def draw_results_enhanced(self, frame, tracks):
        """Enhanced drawing with better visualization"""
        try:
            # Color palette for different classes
            colors = [
                (0, 255, 0),    # Green
                (255, 0, 0),    # Blue
                (0, 0, 255),    # Red
                (255, 255, 0),  # Cyan
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Yellow
                (128, 0, 128),  # Purple
                (255, 165, 0),  # Orange
                (0, 128, 255),  # Light Blue
                (255, 20, 147), # Deep Pink
                (50, 205, 50),  # Lime Green
                (220, 20, 60)   # Crimson
            ]
            
            # Draw tracks
            for track in tracks:
                if len(track) >= 6:
                    x1, y1, x2, y2, track_id, confidence = track[:6]
                else:
                    x1, y1, x2, y2, track_id = track[:5]
                    confidence = 0.0
                
                # Get class info
                if track_id in self.tracker.track_to_class:
                    cls_id = self.tracker.track_to_class[track_id]
                    label = class_names[cls_id]
                    color = colors[cls_id % len(colors)]
                else:
                    label = "Unknown"
                    color = (128, 128, 128)  # Gray
                
                # Draw bounding box with thickness based on confidence
                thickness = max(1, int(confidence * 4))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Prepare labels
                labels = [f"ID:{track_id}"]
                if Config.SHOW_CONFIDENCE:
                    labels.append(f"Conf:{confidence:.2f}")
                if Config.SHOW_CLASS_NAMES:
                    # Show shortened class name for better readability
                    short_label = label[:15] + "..." if len(label) > 15 else label
                    labels.append(short_label)
                
                # Draw labels with background
                label_y = y1 - 10
                for i, label_text in enumerate(labels):
                    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    label_bg_y = label_y - 15 - (i * 20)
                    
                    # Draw background rectangle
                    cv2.rectangle(frame, 
                                (x1, label_bg_y - 2), 
                                (x1 + label_size[0] + 4, label_bg_y + 15),
                                color, -1)
                    
                    # Draw text
                    cv2.putText(frame, label_text, 
                               (x1 + 2, label_bg_y + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Enhanced info display
            self.draw_info_panel(frame)
                           
        except Exception as e:
            logger.error(f"Error drawing results: {e}")
    
    def draw_info_panel(self, frame):
        """Draw enhanced information panel"""
        try:
            total_bags = sum(len(ids) for ids in self.tracker.unique_bags.values())
            stats = self.performance_monitor.get_stats()
            
            # Create semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (500, 200), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Main info
            info_lines = [
                f"Total Bags Detected: {total_bags}",
                f"FPS: {stats['avg_fps']:.1f}",
                f"Processing: {stats['avg_processing_ms']:.1f}ms",
                f"Frames: {self.frame_count} (Processed: {self.processed_frame_count})",
                f"CPU: {stats['avg_cpu_percent']:.1f}% | Memory: {stats['avg_memory_percent']:.1f}%"
            ]
            
            if 'avg_gpu_memory_gb' in stats:
                info_lines.append(f"GPU Memory: {stats['avg_gpu_memory_gb']:.1f}GB")
            
            for i, line in enumerate(info_lines):
                y_pos = 35 + (i * 25)
                cv2.putText(frame, line, (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Per-class counts
            y_offset = 35 + len(info_lines) * 25 + 10
            cv2.putText(frame, "Per Class Count:", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            active_classes = [(name, len(ids)) for name, ids in self.tracker.unique_bags.items() if len(ids) > 0]
            for i, (class_name, count) in enumerate(active_classes[:5]):  # Show top 5
                short_name = class_name[:20] + "..." if len(class_name) > 20 else class_name
                text = f"{short_name}: {count}"
                y_pos = y_offset + 20 + (i * 20)
                cv2.putText(frame, text, (30, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                           
        except Exception as e:
            logger.error(f"Error drawing info panel: {e}")
    
    def run(self):
        """Main processing loop optimized for PC"""
        logger.info("🚀 Starting PC Real-time Bag Counter...")
        
        if not self.load_model_optimized():
            logger.error("Failed to load model")
            return False
            
        if not self.initialize_camera_pc():
            logger.error("Failed to initialize camera")
            return False
        
        logger.info("🎬 Starting real-time processing...")
        logger.info("Controls: 'q' to quit, 's' to save, 'r' to reset counts, 'f' to toggle fullscreen")
        logger.info(f"Auto-save every {Config.SAVE_INTERVAL} seconds")
        
        # Create resizable window
        cv2.namedWindow('PC Real-time Bag Counter', cv2.WINDOW_NORMAL)
        if Config.DISPLAY_SCALE != 1.0:
            display_width = int(Config.FRAME_WIDTH * Config.DISPLAY_SCALE)
            display_height = int(Config.FRAME_HEIGHT * Config.DISPLAY_SCALE)
            cv2.resizeWindow('PC Real-time Bag Counter', display_width, display_height)
        
        fullscreen = False
        
        try:
            consecutive_failures = 0
            max_failures = 10
            
            while self.running:
                ret, frame = self.cap.read()
                
                if not ret or frame is None or frame.size == 0:
                    consecutive_failures += 1
                    logger.warning(f"Frame read failed ({consecutive_failures}/{max_failures})")
                    
                    if consecutive_failures >= max_failures:
                        logger.error("Too many consecutive failures. Attempting camera restart...")
                        if not self.initialize_camera_pc():
                            logger.error("Camera restart failed. Shutting down.")
                            break
                        consecutive_failures = 0
                    
                    time.sleep(0.1)
                    continue
                
                consecutive_failures = 0
                self.frame_count += 1
                
                # Process frames based on skip setting
                if self.frame_count % (Config.SKIP_FRAMES + 1) == 0:
                    processed_frame = self.process_frame_optimized(frame.copy())
                    self.processed_frame_count += 1
                else:
                    processed_frame = frame.copy()
                    # Still show basic info on skipped frames
                    self.draw_info_panel(processed_frame)
                
                # Display frame
                try:
                    cv2.imshow('PC Real-time Bag Counter', processed_frame)
                except Exception as e:
                    logger.warning(f"Display error: {e}")
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("🛑 Quit requested")
                    break
                elif key == ord('s'):
                    logger.info("💾 Manual save requested")
                    stats = self.performance_monitor.get_stats()
                    confidence_data = self.tracker.track_to_confidence
                    self.excel_saver.save_results(
                        self.tracker.unique_bags, confidence_data, "_manual", stats
                    )
                elif key == ord('r'):
                    logger.info("🔄 Resetting counts")
                    self.tracker.unique_bags = {name: set() for name in class_names}
                    self.tracker.track_to_class = {}
                    self.tracker.track_to_confidence = {}
                    logger.info("✅ Counts reset")
                elif key == ord('f'):
                    # Toggle fullscreen
                    fullscreen = not fullscreen
                    if fullscreen:
                        cv2.setWindowProperty('PC Real-time Bag Counter', 
                                            cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty('PC Real-time Bag Counter', 
                                            cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    logger.info(f"Fullscreen: {'ON' if fullscreen else 'OFF'}")
                elif key == 27:  # ESC key
                    logger.info("🛑 ESC pressed - quitting")
                    break
                
                # Auto-save
                if time.time() - self.last_save_time > Config.SAVE_INTERVAL:
                    stats = self.performance_monitor.get_stats()
                    confidence_data = self.tracker.track_to_confidence
                    self.excel_saver.save_results(
                        self.tracker.unique_bags, confidence_data, "_auto", stats
                    )
                    self.last_save_time = time.time()
                
                # Statistics logging
                if self.processed_frame_count % 100 == 0 and self.processed_frame_count > 0:
                    total_bags = sum(len(ids) for ids in self.tracker.unique_bags.values())
                    stats = self.performance_monitor.get_stats()
                    logger.info(f"📊 Frame {self.frame_count}: {total_bags} bags, "
                               f"{stats['avg_fps']:.1f} FPS, {stats['avg_processing_ms']:.1f}ms, "
                               f"CPU: {stats['avg_cpu_percent']:.1f}%")
        
        except Exception as e:
            logger.error(f"❌ Error during processing: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
        
        return True
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up...")
        
        # Final save
        stats = self.performance_monitor.get_stats()
        confidence_data = self.tracker.track_to_confidence
        self.excel_saver.save_results(
            self.tracker.unique_bags, confidence_data, "_final", stats
        )
        
        # Release resources
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Clear GPU memory if used
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Print final statistics
        total_bags = sum(len(ids) for ids in self.tracker.unique_bags.values())
        logger.info(f"\n📋 FINAL STATISTICS:")
        logger.info(f"   Total frames captured: {self.frame_count}")
        logger.info(f"   Total frames processed: {self.processed_frame_count}")
        logger.info(f"   Processing efficiency: {(self.processed_frame_count/self.frame_count*100):.1f}%")
        logger.info(f"   Total bags detected: {total_bags}")
        logger.info(f"   Results saved to: {Config.OUTPUT_DIR}")
        
        final_stats = self.performance_monitor.get_stats()
        logger.info(f"   Average FPS: {final_stats['avg_fps']:.1f}")
        logger.info(f"   Average processing time: {final_stats['avg_processing_ms']:.1f}ms")
        logger.info(f"   Average CPU usage: {final_stats['avg_cpu_percent']:.1f}%")
        logger.info(f"   Average memory usage: {final_stats['avg_memory_percent']:.1f}%")
        
        if 'avg_gpu_memory_gb' in final_stats:
            logger.info(f"   Average GPU memory: {final_stats['avg_gpu_memory_gb']:.1f}GB")
        
        # Per-class summary
        logger.info("\n📦 PER-CLASS SUMMARY:")
        for name, ids in self.tracker.unique_bags.items():
            if len(ids) > 0:
                avg_conf = np.mean([confidence_data.get(id, 0) for id in ids if id in confidence_data])
                logger.info(f"   {name}: {len(ids)} bags (avg conf: {avg_conf:.2f})")

# ===================== DIAGNOSTIC FUNCTIONS =====================
def run_pc_diagnostics():
    """Run comprehensive PC diagnostics"""
    logger.info("🔍 Running PC diagnostics...")
    
    # System information
    try:
        import platform
        logger.info(f"\n💻 System Information:")
        logger.info(f"   OS: {platform.system()} {platform.release()}")
        logger.info(f"   Processor: {platform.processor()}")
        logger.info(f"   Python: {platform.python_version()}")
        logger.info(f"   Architecture: {platform.architecture()[0]}")
    except Exception as e:
        logger.warning(f"Could not get system info: {e}")
    
    # Hardware information
    try:
        logger.info(f"\n🔧 Hardware Information:")
        logger.info(f"   CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
        memory = psutil.virtual_memory()
        logger.info(f"   RAM: {memory.total / (1024**3):.1f}GB total, {memory.available / (1024**3):.1f}GB available")
        
        # GPU information
        if torch.cuda.is_available():
            logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"   CUDA version: {torch.version.cuda}")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"   GPU memory: {gpu_memory:.1f}GB")
        else:
            logger.info("   GPU: Not available or not CUDA-compatible")
    except Exception as e:
        logger.warning(f"Could not get hardware info: {e}")
    
    # OpenCV information
    try:
        logger.info(f"\n📹 OpenCV Information:")
        logger.info(f"   Version: {cv2.__version__}")
        build_info = cv2.getBuildInformation()
        logger.info(f"   DirectShow support: {'YES' if 'DirectShow' in build_info else 'NO'}")
        logger.info(f"   CUDA support: {'YES' if 'CUDA' in build_info else 'NO'}")
        logger.info(f"   FFmpeg support: {'YES' if 'FFMPEG' in build_info else 'NO'}")
    except Exception as e:
        logger.warning(f"Could not get OpenCV info: {e}")
    
    # Camera testing
    logger.info(f"\n📷 Camera Testing:")
    available_cameras = PCCameraUtils.get_available_cameras()
    logger.info(f"   Available cameras: {available_cameras}")
    
    for camera_id in available_cameras[:3]:  # Test first 3 cameras
        logger.info(f"\n   Testing camera {camera_id}:")
        capabilities = PCCameraUtils.get_camera_capabilities(camera_id)
        if capabilities:
            logger.info(f"     Current resolution: {capabilities['current_width']}x{capabilities['current_height']}")
            logger.info(f"     FPS: {capabilities['fps']}")
            logger.info(f"     Backend: {capabilities['backend']}")
            logger.info(f"     Supported resolutions: {capabilities['supported_resolutions']}")
        else:
            logger.warning(f"     Could not get capabilities for camera {camera_id}")
    
    # Model testing
    logger.info(f"\n🤖 Model Testing:")
    if os.path.exists(Config.MODEL_PATH):
        model_size = os.path.getsize(Config.MODEL_PATH) / (1024**2)
        logger.info(f"   Model found: {Config.MODEL_PATH}")
        logger.info(f"   Model size: {model_size:.1f}MB")
        
        try:
            # Test model loading
            test_model = YOLO(Config.MODEL_PATH)
            logger.info(f"   ✅ Model loads successfully")
            
            # Test inference
            dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
            start_time = time.time()
            results = test_model(dummy_input, verbose=False)
            inference_time = (time.time() - start_time) * 1000
            logger.info(f"   ✅ Inference test successful: {inference_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"   ❌ Model test failed: {e}")
    else:
        logger.error(f"   ❌ Model not found: {Config.MODEL_PATH}")

# ===================== MAIN EXECUTION =====================
if __name__ == "__main__":
    print("🖥️ PC Real-time Bag Counter")
    print("=" * 50)
    
    # Check for diagnostic mode
    if len(sys.argv) > 1 and sys.argv[1] == "--diagnostics":
        run_pc_diagnostics()
        sys.exit(0)
    
    # Check dependencies
    try:
        import ultralytics
        logger.info(f"✅ Ultralytics YOLO version: {ultralytics.__version__}")
    except ImportError:
        logger.error("❌ Ultralytics not found. Please install: pip install ultralytics")
        sys.exit(1)
    
    counter = PCRealTimeBagCounter()
    
    try:
        success = counter.run()
        if success:
            logger.info("✅ Application completed successfully")
        else:
            logger.error("❌ Application failed")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n🔄 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        logger.info("👋 Goodbye!")
        sys.exit(0)