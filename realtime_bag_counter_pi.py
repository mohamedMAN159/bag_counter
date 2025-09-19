# realtime_bag_counter_pi_camera_fixed.py
# Enhanced version with robust camera initialization for Raspberry Pi

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
import subprocess

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

# ===================== RASPBERRY PI OPTIMIZATIONS =====================
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# ===================== CONFIGURATION =====================
class Config:
    # File paths - CHANGE THESE FOR YOUR SETUP
    MODEL_PATH = r"D:\ready\bag_counter\yolo_model\best.pt"
    CAMERA_INDEX = 0
    
    # Video settings optimized for Pi 5
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS = 15
    
    # Detection settings
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    
    # Tracking settings
    TRACK_DISTANCE_THRESHOLD = 100
    TRACK_MIN_HITS = 3
    TRACK_MAX_AGE = 30
    
    # Performance settings
    SKIP_FRAMES = 2  # Process every nth frame for better performance
    MAX_DETECTIONS = 50  # Limit max detections per frame
    
    # Camera retry settings
    CAMERA_RETRY_COUNT = 5
    CAMERA_RETRY_DELAY = 2
    CAMERA_WARMUP_FRAMES = 10  # Number of frames to skip for camera warmup
    
    # Save settings
    SAVE_INTERVAL = 60
    OUTPUT_DIR = r"D:\ready\bag_counter\bag_counts"

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

# ===================== CAMERA UTILITIES =====================
class CameraUtils:
    @staticmethod
    def check_camera_devices():
        """Check available camera devices"""
        available_devices = []
        
        # Check /dev/video* devices
        for i in range(10):  # Check first 10 devices
            device_path = f"/dev/video{i}"
            if os.path.exists(device_path):
                available_devices.append(i)
        
        logger.info(f"Available camera devices: {available_devices}")
        return available_devices
    
    @staticmethod
    def check_raspicam():
        """Check if Raspberry Pi camera is enabled"""
        try:
            # Check if libcamera or raspistill is available
            result = subprocess.run(['libcamera-hello', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info("✅ libcamera detected")
                return True
        except:
            pass
        
        try:
            result = subprocess.run(['raspistill', '--help'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info("✅ raspistill detected")
                return True
        except:
            pass
        
        logger.warning("⚠️ No Raspberry Pi camera tools detected")
        return False
    
    @staticmethod
    def get_camera_info(index):
        """Get camera information"""
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            return None
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'backend': cap.getBackendName()
        }
        
        cap.release()
        return info

# ===================== IMPROVED TRACKER CLASS =====================
class ImprovedTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.unique_bags = {name: set() for name in class_names}
        self.track_to_class = {}
        self.frame_count = 0
        
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
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
        """Update tracker with new detections using improved matching."""
        self.frame_count += 1
        
        if len(detections) == 0:
            # Age existing tracks
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > Config.TRACK_MAX_AGE:
                    del self.tracks[track_id]
            return []
        
        # Limit detections for performance
        if len(detections) > Config.MAX_DETECTIONS:
            detections = sorted(detections, key=lambda x: x[4], reverse=True)[:Config.MAX_DETECTIONS]
        
        # Match detections to existing tracks
        matched_tracks = []
        unmatched_detections = list(range(len(detections)))
        
        # Create cost matrix
        cost_matrix = []
        track_ids = list(self.tracks.keys())
        
        for track_id in track_ids:
            if self.tracks[track_id]['age'] > Config.TRACK_MAX_AGE:
                continue
                
            track_costs = []
            track_box = self.tracks[track_id]['bbox']
            
            for i, det in enumerate(detections):
                if i not in unmatched_detections:
                    track_costs.append(float('inf'))
                    continue
                    
                det_box = det[:4]
                
                # Calculate distance cost
                track_center = [(track_box[0] + track_box[2]) / 2, (track_box[1] + track_box[3]) / 2]
                det_center = [(det_box[0] + det_box[2]) / 2, (det_box[1] + det_box[3]) / 2]
                distance = np.sqrt((track_center[0] - det_center[0])**2 + (track_center[1] - det_center[1])**2)
                
                # Calculate IoU cost
                iou = self.calculate_iou(track_box, det_box)
                
                # Combined cost (lower is better)
                if distance < Config.TRACK_DISTANCE_THRESHOLD and iou > 0.1:
                    cost = distance * (1 - iou)  # Weight by inverse IoU
                else:
                    cost = float('inf')
                
                track_costs.append(cost)
            
            cost_matrix.append(track_costs)
        
        # Simple greedy matching
        for i, track_id in enumerate(track_ids):
            if self.tracks[track_id]['age'] > Config.TRACK_MAX_AGE:
                continue
                
            if i >= len(cost_matrix):
                continue
                
            costs = cost_matrix[i]
            if not costs or min(costs) == float('inf'):
                continue
                
            best_match = np.argmin(costs)
            if best_match in unmatched_detections and costs[best_match] < float('inf'):
                # Update existing track
                det = detections[best_match]
                x1, y1, x2, y2, conf, cls_id = det
                
                self.tracks[track_id].update({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'class_id': int(cls_id),
                    'age': 0,
                    'hits': self.tracks[track_id]['hits'] + 1,
                    'last_seen': self.frame_count
                })
                
                matched_tracks.append(track_id)
                unmatched_detections.remove(best_match)
        
        # Create new tracks for unmatched detections
        for i in unmatched_detections:
            det = detections[i]
            x1, y1, x2, y2, conf, cls_id = det
            
            self.tracks[self.next_id] = {
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': conf,
                'class_id': int(cls_id),
                'age': 0,
                'hits': 1,
                'last_seen': self.frame_count
            }
            
            self.next_id += 1
        
        # Return confirmed tracks and update unique bags
        confirmed_tracks = []
        for track_id, track in self.tracks.items():
            if track['hits'] >= Config.TRACK_MIN_HITS and track['age'] == 0:
                confirmed_tracks.append([
                    track['bbox'][0], track['bbox'][1], 
                    track['bbox'][2], track['bbox'][3], 
                    track_id
                ])
                
                # Add to unique bags
                cls_id = track['class_id']
                if cls_id < len(class_names):
                    label = class_names[cls_id]
                    if track_id not in self.track_to_class:
                        self.track_to_class[track_id] = cls_id
                        self.unique_bags[label].add(track_id)
                        logger.info(f"🆕 New bag: ID {track_id} - {label}")
        
        return confirmed_tracks

# ===================== PERFORMANCE MONITOR =====================
class PerformanceMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.processing_times = []
        self.fps_times = []
        self.last_time = time.time()
        
    def update(self, processing_time):
        current_time = time.time()
        
        # Update processing times
        self.processing_times.append(processing_time)
        if len(self.processing_times) > self.window_size:
            self.processing_times.pop(0)
        
        # Update FPS
        self.fps_times.append(current_time - self.last_time)
        if len(self.fps_times) > self.window_size:
            self.fps_times.pop(0)
        
        self.last_time = current_time
    
    def get_stats(self):
        if not self.processing_times or not self.fps_times:
            return {"avg_processing_ms": 0, "avg_fps": 0}
        
        avg_processing = np.mean(self.processing_times) * 1000  # ms
        avg_fps = 1.0 / np.mean(self.fps_times) if np.mean(self.fps_times) > 0 else 0
        
        return {
            "avg_processing_ms": avg_processing,
            "avg_fps": avg_fps,
            "min_processing_ms": min(self.processing_times) * 1000,
            "max_processing_ms": max(self.processing_times) * 1000
        }

# ===================== EXCEL SAVER =====================
class ExcelSaver:
    def __init__(self):
        self.output_dir = Path(Config.OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_results(self, unique_bags, suffix="", performance_stats=None):
        """Save results to Excel with timestamp and performance stats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bag_count_{timestamp}{suffix}.xlsx"
        filepath = self.output_dir / filename
        
        try:
            workbook = xlsxwriter.Workbook(str(filepath), {'nan_inf_to_errors': True})
            worksheet = workbook.add_worksheet('عدد الأكياس')
            
            # Headers
            headers = ['النوع', 'العدد', 'معرفات الأكياس', 'وقت الحفظ']
            for col, header in enumerate(headers):
                worksheet.write(0, col, header)
            
            row = 1
            total_count = 0
            
            for name, ids in unique_bags.items():
                count = len(ids)
                if count > 0:
                    ids_str = ", ".join(map(str, sorted(ids)))
                    worksheet.write(row, 0, name)
                    worksheet.write(row, 1, count)
                    worksheet.write(row, 2, ids_str)
                    worksheet.write(row, 3, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    total_count += count
                    row += 1
            
            # Total row
            worksheet.write(row, 0, 'المجموع الكلي')
            worksheet.write(row, 1, total_count)
            worksheet.write(row, 3, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            # Performance stats sheet
            if performance_stats:
                perf_sheet = workbook.add_worksheet('Performance')
                perf_sheet.write(0, 0, 'Metric')
                perf_sheet.write(0, 1, 'Value')
                
                row = 1
                for key, value in performance_stats.items():
                    perf_sheet.write(row, 0, key)
                    perf_sheet.write(row, 1, str(value))
                    row += 1
            
            workbook.close()
            
            logger.info(f"💾 Results saved: {filename} (Total: {total_count} bags)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving Excel: {e}")
            return False

# ===================== MAIN APPLICATION WITH IMPROVED CAMERA HANDLING =====================
class RealTimeBagCounter:
    def __init__(self):
        self.running = True
        self.model = None
        self.cap = None
        self.tracker = ImprovedTracker()
        self.excel_saver = ExcelSaver()
        self.performance_monitor = PerformanceMonitor()
        self.last_save_time = time.time()
        self.frame_count = 0
        self.processed_frame_count = 0
        self.warmup_frame_count = 0
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"🔄 Received signal {signum}. Shutting down gracefully...")
        self.running = False
    
    def load_model_safe(self):
        """Load YOLO model with PyTorch 2.6 compatibility"""
        logger.info(f"📁 Model path: {Config.MODEL_PATH}")
        
        if not os.path.exists(Config.MODEL_PATH):
            logger.error(f"❌ Model not found: {Config.MODEL_PATH}")
            return False
        
        logger.info(f"📁 Model file size: {os.path.getsize(Config.MODEL_PATH) / (1024*1024):.1f} MB")
        
        try:
            logger.info("🔥 Attempting to load model with safe globals...")
            self.model = YOLO(Config.MODEL_PATH)
            logger.info("✅ Model loaded successfully with safe globals")
            return True
            
        except Exception as e1:
            logger.warning(f"⚠️ Safe loading failed: {str(e1)[:200]}...")
            
            try:
                logger.info("🔥 Attempting to load with weights_only=False...")
                import torch
                original_load = torch.load
                
                def patched_load(*args, **kwargs):
                    if 'weights_only' not in kwargs:
                        kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                
                torch.load = patched_load
                
                try:
                    self.model = YOLO(Config.MODEL_PATH)
                    logger.info("✅ Model loaded with weights_only=False")
                    return True
                finally:
                    torch.load = original_load
                
            except Exception as e2:
                logger.error(f"❌ Alternative loading failed: {str(e2)[:200]}...")
                return False
    
    def initialize_camera_robust(self):
        """Initialize camera with robust error handling and multiple attempts"""
        logger.info(f"📹 Initializing camera...")
        
        # Check available cameras
        available_cameras = CameraUtils.check_camera_devices()
        if not available_cameras:
            logger.error("❌ No camera devices found in /dev/video*")
            return False
        
        # Check Raspberry Pi camera tools
        CameraUtils.check_raspicam()
        
        # Try different camera indices
        camera_indices_to_try = [Config.CAMERA_INDEX] + [i for i in available_cameras if i != Config.CAMERA_INDEX]
        
        # Try different backends
        backends_to_try = [
            (cv2.CAP_V4L2, "CAP_V4L2"),
            (cv2.CAP_GSTREAMER, "CAP_GSTREAMER"),
            (cv2.CAP_ANY, "CAP_ANY")
        ]
        
        for camera_index in camera_indices_to_try:
            logger.info(f"🎯 Trying camera index: {camera_index}")
            
            for backend, backend_name in backends_to_try:
                logger.info(f"🔄 Trying backend: {backend_name}")
                
                for retry in range(Config.CAMERA_RETRY_COUNT):
                    try:
                        # Release any existing capture
                        if self.cap:
                            self.cap.release()
                            time.sleep(0.5)
                        
                        # Create new capture
                        self.cap = cv2.VideoCapture(camera_index, backend)
                        
                        if not self.cap.isOpened():
                            logger.warning(f"Failed to open camera {camera_index} with {backend_name} (attempt {retry + 1})")
                            time.sleep(Config.CAMERA_RETRY_DELAY)
                            continue
                        
                        logger.info(f"✅ Camera opened: index {camera_index}, backend {backend_name}")
                        
                        # Set camera properties with error handling
                        properties = [
                            (cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH),
                            (cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT),
                            (cv2.CAP_PROP_FPS, Config.FPS),
                            (cv2.CAP_PROP_BUFFERSIZE, 1)
                        ]
                        
                        for prop, value in properties:
                            try:
                                self.cap.set(prop, value)
                            except Exception as e:
                                logger.warning(f"Failed to set property {prop}: {e}")
                        
                        # Get actual settings
                        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                        
                        logger.info(f"📐 Camera configured: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
                        
                        # Test frame capture with multiple attempts
                        frame_captured = False
                        for frame_attempt in range(5):
                            ret, test_frame = self.cap.read()
                            if ret and test_frame is not None and test_frame.size > 0:
                                logger.info(f"✅ Test frame captured: {test_frame.shape}")
                                frame_captured = True
                                break
                            else:
                                logger.warning(f"Frame capture attempt {frame_attempt + 1} failed")
                                time.sleep(0.5)
                        
                        if frame_captured:
                            # Success! Skip some initial frames for camera warmup
                            logger.info(f"🔥 Camera warmup: skipping {Config.CAMERA_WARMUP_FRAMES} frames...")
                            for _ in range(Config.CAMERA_WARMUP_FRAMES):
                                self.cap.read()
                            
                            logger.info("✅ Camera initialization successful!")
                            return True
                        else:
                            logger.warning("❌ Failed to capture test frames")
                            
                    except Exception as e:
                        logger.warning(f"Camera initialization error (attempt {retry + 1}): {e}")
                        time.sleep(Config.CAMERA_RETRY_DELAY)
                        continue
        
        logger.error("❌ Failed to initialize any camera after all attempts")
        return False
    
    def process_frame(self, frame):
        """Process a single frame with performance monitoring"""
        start_time = time.time()
        
        try:
            # Run YOLO detection
            results = self.model(
                frame, 
                conf=Config.CONFIDENCE_THRESHOLD,
                iou=Config.IOU_THRESHOLD,
                verbose=False
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
                        logger.warning(f"Error processing detection box: {e}")
                        continue
            
            # Update tracker
            tracks = self.tracker.update(detections)
            
            # Draw results
            self.draw_results(frame, tracks)
            
            # Update performance monitoring
            processing_time = time.time() - start_time
            self.performance_monitor.update(processing_time)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error in process_frame: {e}")
            return frame
    
    def draw_results(self, frame, tracks):
        """Draw tracking results on frame"""
        try:
            # Draw tracks
            for track in tracks:
                x1, y1, x2, y2, track_id = track
                
                # Get class info
                if track_id in self.tracker.track_to_class:
                    cls_id = self.tracker.track_to_class[track_id]
                    label = class_names[cls_id]
                    color = (0, 255, 0)  # Green
                else:
                    label = "Unknown"
                    color = (0, 0, 255)  # Red
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, label[:20], (x1, y1-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Display performance info
            total_bags = sum(len(ids) for ids in self.tracker.unique_bags.values())
            stats = self.performance_monitor.get_stats()
            
            info_lines = [
                f"Total Bags: {total_bags}",
                f"Processing: {stats['avg_processing_ms']:.1f}ms",
                f"FPS: {stats['avg_fps']:.1f}",
                f"Frame: {self.frame_count} (Processed: {self.processed_frame_count})"
            ]
            
            for i, line in enumerate(info_lines):
                y_pos = 30 + (i * 25)
                cv2.putText(frame, line, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                           
        except Exception as e:
            logger.error(f"Error drawing results: {e}")
    
    def run(self):
        """Main processing loop with improved error handling"""
        if not self.load_model_safe():
            logger.error("Failed to load model")
            return False
            
        if not self.initialize_camera_robust():
            logger.error("Failed to initialize camera")
            return False
        
        logger.info("🎬 Starting real-time processing...")
        logger.info("Press 'q' to quit, 's' to save manually")
        logger.info(f"Auto-save every {Config.SAVE_INTERVAL} seconds")
        
        try:
            consecutive_failures = 0
            max_failures = 10
            
            while self.running:
                ret, frame = self.cap.read()
                
                if not ret or frame is None or frame.size == 0:
                    consecutive_failures += 1
                    logger.warning(f"Frame read failed ({consecutive_failures}/{max_failures})")
                    
                    if consecutive_failures >= max_failures:
                        logger.error("Too many consecutive frame failures. Attempting camera reset...")
                        if not self.initialize_camera_robust():
                            logger.error("Camera reset failed. Shutting down.")
                            break
                        consecutive_failures = 0
                    
                    time.sleep(0.1)
                    continue
                
                consecutive_failures = 0  # Reset on successful frame
                self.frame_count += 1
                
                # Skip frames for performance
                if self.frame_count % (Config.SKIP_FRAMES + 1) == 0:
                    processed_frame = self.process_frame(frame.copy())
                    self.processed_frame_count += 1
                else:
                    processed_frame = frame.copy()
                    # Still draw basic info on skipped frames
                    total_bags = sum(len(ids) for ids in self.tracker.unique_bags.values())
                    cv2.putText(processed_frame, f"Total Bags: {total_bags}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Show frame
                try:
                    cv2.imshow('Real-time Bag Counter - Raspberry Pi', processed_frame)
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
                    self.excel_saver.save_results(self.tracker.unique_bags, "_manual", stats)
                
                # Auto-save every interval
                if time.time() - self.last_save_time > Config.SAVE_INTERVAL:
                    stats = self.performance_monitor.get_stats()
                    self.excel_saver.save_results(self.tracker.unique_bags, "_auto", stats)
                    self.last_save_time = time.time()
                
                # Print statistics every 100 processed frames
                if self.processed_frame_count % 100 == 0 and self.processed_frame_count > 0:
                    total_bags = sum(len(ids) for ids in self.tracker.unique_bags.values())
                    stats = self.performance_monitor.get_stats()
                    logger.info(f"📊 Frame {self.frame_count} (Processed: {self.processed_frame_count}): "
                               f"{total_bags} bags, {stats['avg_fps']:.1f} FPS, "
                               f"{stats['avg_processing_ms']:.1f}ms avg processing")
        
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
        self.excel_saver.save_results(self.tracker.unique_bags, "_final", stats)
        
        # Release resources
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        total_bags = sum(len(ids) for ids in self.tracker.unique_bags.values())
        logger.info(f"\n📋 FINAL STATISTICS:")
        logger.info(f"   Total frames captured: {self.frame_count}")
        logger.info(f"   Total frames processed: {self.processed_frame_count}")
        logger.info(f"   Total bags detected: {total_bags}")
        logger.info(f"   Results saved to: {Config.OUTPUT_DIR}")
        logger.info(f"   Final performance: {stats}")
        
        for name, ids in self.tracker.unique_bags.items():
            if len(ids) > 0:
                logger.info(f"   📦 {name}: {len(ids)} bags")

# ===================== DIAGNOSTIC FUNCTIONS =====================
def run_camera_diagnostics():
    """Run comprehensive camera diagnostics"""
    logger.info("🔍 Running camera diagnostics...")
    
    # Check available devices
    available_devices = CameraUtils.check_camera_devices()
    logger.info(f"Available video devices: {available_devices}")
    
    # Check Raspberry Pi camera
    CameraUtils.check_raspicam()
    
    # Test each available device
    for device_id in available_devices:
        logger.info(f"\n🎯 Testing device {device_id}:")
        info = CameraUtils.get_camera_info(device_id)
        if info:
            logger.info(f"   Resolution: {info['width']}x{info['height']}")
            logger.info(f"   FPS: {info['fps']}")
            logger.info(f"   Backend: {info['backend']}")
            
            # Try to capture a frame
            cap = cv2.VideoCapture(device_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    logger.info(f"   ✅ Frame capture successful: {frame.shape}")
                else:
                    logger.warning(f"   ❌ Frame capture failed")
                cap.release()
            else:
                logger.warning(f"   ❌ Could not open device {device_id}")
        else:
            logger.warning(f"   ❌ Could not get info for device {device_id}")
    
    # Check system info
    try:
        import platform
        logger.info(f"\n💻 System info:")
        logger.info(f"   OS: {platform.system()} {platform.release()}")
        logger.info(f"   Python: {platform.python_version()}")
        logger.info(f"   OpenCV: {cv2.__version__}")
        
        # Check GPU/hardware acceleration
        logger.info(f"   OpenCV build info:")
        logger.info(f"     - With GSTREAMER: {'YES' if cv2.getBuildInformation().find('GStreamer') != -1 else 'NO'}")
        logger.info(f"     - With V4L/V4L2: {'YES' if cv2.getBuildInformation().find('V4L') != -1 else 'NO'}")
        
    except Exception as e:
        logger.warning(f"Could not get system info: {e}")

# ===================== MAIN EXECUTION =====================
if __name__ == "__main__":
    print("🖥️ Raspberry Pi 5 - Real-time Bag Counter (Camera Fixed)")
    print("="*60)
    
    # Check if diagnostic mode is requested
    if len(sys.argv) > 1 and sys.argv[1] == "--diagnostics":
        run_camera_diagnostics()
        sys.exit(0)
    
    counter = RealTimeBagCounter()
    
    try:
        success = counter.run()
        if success:
            logger.info("✅ Application completed successfully")
        else:
            logger.error("❌ Application failed to initialize")
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