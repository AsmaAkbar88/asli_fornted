---
sidebar_label: 'Object Detection Integration'
---

# Object Detection Integration

This section covers the integration of object detection capabilities into the Physical AI system, enabling the robot to identify, localize, and classify objects in its environment. Object detection serves as a critical bridge between raw sensor data and actionable information for manipulation and navigation tasks.

## Object Detection Architecture Overview
 
The object detection system in our integrated Physical AI system follows a multi-modal approach:

```
RGB Camera → Feature Extraction → Object Detection → 3D Localization → Object Database → Action Interface
Thermal/Depth → ──────────────────────────────────────────────────────────────────────────────────
```

This architecture enables robust object detection across various lighting conditions and environments.

## 2D Object Detection

### YOLO-based Detector:
```python
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

class YOLOObjectDetector:
    def __init__(self, model_path=None, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold

        # Load pre-trained YOLO model or create a custom one
        if model_path:
            self.model = torch.load(model_path)
        else:
            # Using a standard model as placeholder
            self.model = self.create_yolo_model()

        self.model.eval()

        # COCO dataset class names (80 classes)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((416, 416)),
            transforms.ToTensor()
        ])

    def create_yolo_model(self):
        """
        Create a YOLO-style model (placeholder implementation)
        In practice, use torch.hub.load('ultralytics/yolov5', 'yolov5s') or similar
        """
        # This is a placeholder - in practice you'd load a real YOLO model
        class DummyYOLO(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Placeholder model
                self.dummy_param = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x):
                # Placeholder output simulating YOLO detections
                # Format: [batch, num_detections, 6] -> [x1, y1, x2, y2, confidence, class_id]
                batch_size = x.shape[0]
                dummy_detections = torch.zeros(batch_size, 100, 6)  # 100 max detections
                return dummy_detections

        return DummyYOLO()

    def detect_objects(self, image):
        """
        Detect objects in an input image
        """
        # Convert image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            detections = self.model(input_tensor)

        # Post-process detections
        results = self.post_process_detections(detections, image.size)

        return results

    def post_process_detections(self, detections, image_size):
        """
        Post-process raw detections to extract meaningful results
        """
        results = []

        # For each detection in the batch (assuming batch size 1)
        for detection in detections:
            # Filter by confidence threshold
            conf_mask = detection[:, 4] > self.confidence_threshold

            if conf_mask.any():
                filtered_detections = detection[conf_mask]

                for det in filtered_detections:
                    x1, y1, x2, y2, conf, class_id = det
                    class_id = int(class_id)

                    # Convert to image coordinates
                    x1 = float(x1) * image_size[0] / 416.0
                    y1 = float(y1) * image_size[1] / 416.0
                    x2 = float(x2) * image_size[0] / 416.0
                    y2 = float(y2) * image_size[1] / 416.0

                    # Create result object
                    result = {
                        'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],  # x, y, width, height
                        'confidence': float(conf),
                        'class_id': class_id,
                        'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f'unknown_{class_id}',
                        'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                    }

                    results.append(result)

        return results

    def visualize_detections(self, image, detections, show_labels=True):
        """
        Visualize object detections on the image
        """
        img_with_boxes = np.array(image.copy())

        for det in detections:
            bbox = det['bbox']
            label = f"{det['class_name']}: {det['confidence']:.2f}" if show_labels else ""

            # Draw bounding box
            cv2.rectangle(
                img_with_boxes,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                (0, 255, 0),
                2
            )

            # Draw label
            if show_labels:
                cv2.putText(
                    img_with_boxes,
                    label,
                    (int(bbox[0]), int(bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        return img_with_boxes
```

### Custom Object Detector for Robotics:
```python
class RoboticsObjectDetector:
    def __init__(self):
        # Pre-trained base detector
        self.base_detector = YOLOObjectDetector()

        # Fine-tuned models for specific robotic objects
        self.specialized_detectors = {}

        # Robot-specific object classes
        self.robotic_objects = [
            'cup', 'bottle', 'box', 'book', 'phone', 'laptop',
            'remote', 'toy', 'utensil', 'container', 'tool'
        ]

        # Object properties database
        self.object_properties = self.initialize_object_properties()

    def initialize_object_properties(self):
        """
        Initialize known object properties for robotic manipulation
        """
        return {
            'cup': {
                'manipulable': True,
                'grasp_points': ['handle', 'rim'],
                'orientation_sensitive': False,
                'weight_range': [0.1, 0.5],  # kg
                'size_range': [0.05, 0.15]  # meters
            },
            'bottle': {
                'manipulable': True,
                'grasp_points': ['neck', 'body'],
                'orientation_sensitive': True,
                'weight_range': [0.2, 2.0],
                'size_range': [0.05, 0.3]
            },
            'book': {
                'manipulable': True,
                'grasp_points': ['spine', 'edge'],
                'orientation_sensitive': False,
                'weight_range': [0.2, 2.0],
                'size_range': [0.1, 0.3]
            },
            'box': {
                'manipulable': True,
                'grasp_points': ['center', 'edges'],
                'orientation_sensitive': False,
                'weight_range': [0.1, 5.0],
                'size_range': [0.05, 0.5]
            }
        }

    def detect_robotic_objects(self, image):
        """
        Detect objects relevant for robotic manipulation
        """
        # Run base detection
        all_detections = self.base_detector.detect_objects(image)

        # Filter for robotic objects
        robotic_detections = []
        for det in all_detections:
            if det['class_name'] in self.robotic_objects:
                # Add object properties
                det['properties'] = self.object_properties.get(det['class_name'], {})
                robotic_detections.append(det)

        return robotic_detections

    def detect_specific_object(self, image, target_object):
        """
        Detect a specific object type with higher precision
        """
        # This would use a specialized detector for the target object
        # For now, filter from general detections
        all_detections = self.base_detector.detect_objects(image)

        target_detections = [
            det for det in all_detections
            if det['class_name'] == target_object
        ]

        return target_detections

    def get_object_grasp_points(self, detection):
        """
        Get recommended grasp points for an object
        """
        class_name = detection['class_name']
        properties = self.object_properties.get(class_name, {})
        grasp_points = properties.get('grasp_points', ['center'])

        # Calculate grasp point coordinates based on bounding box
        bbox = detection['bbox']
        x, y, w, h = bbox

        grasp_coords = []
        for point in grasp_points:
            if point == 'center':
                grasp_coords.append([x + w/2, y + h/2])
            elif point == 'handle':
                # Approximate handle location for cups
                grasp_coords.append([x + w/4, y + h/2])
            elif point == 'rim':
                # Approximate rim location
                grasp_coords.append([x + w/2, y + h/10])
            elif point == 'neck':
                # Approximate neck location for bottles
                grasp_coords.append([x + w/2, y + h/4])
            elif point == 'spine':
                # Book spine
                grasp_coords.append([x + w/10, y + h/2])

        return grasp_coords

    def estimate_object_pose(self, image, detection):
        """
        Estimate 6D pose of an object (simplified for 2D detection)
        """
        # In a real implementation, this would use RGB-D data or specialized pose estimation
        # For now, return a simplified pose based on 2D detection

        bbox = detection['bbox']
        center_x, center_y = detection['center']

        # Simplified pose estimation
        pose = {
            'position': {
                'x': center_x,
                'y': center_y,
                'z': 0.0  # Depth would come from depth sensor
            },
            'orientation': {
                'roll': 0.0,
                'pitch': 0.0,
                'yaw': 0.0  # Orientation estimation would require more info
            },
            'confidence': detection['confidence']
        }

        return pose
```

## 3D Object Detection and Localization

### RGB-D Integration:
```python
class RGBDObjectDetector:
    def __init__(self, intrinsic_matrix):
        self.intrinsic_matrix = intrinsic_matrix  # Camera intrinsic parameters
        self.detector_2d = YOLOObjectDetector()

    def detect_and_localize(self, rgb_image, depth_image):
        """
        Detect objects in RGB and localize them in 3D using depth
        """
        # Get 2D detections
        detections_2d = self.detector_2d.detect_objects(rgb_image)

        # For each 2D detection, get 3D position from depth
        detections_3d = []
        for det in detections_2d:
            bbox = det['bbox']

            # Extract depth information from bounding box region
            x, y, w, h = [int(coord) for coord in bbox]
            x_center, y_center = int(det['center'][0]), int(det['center'][1])

            # Get depth at center of bounding box (or average over region)
            if (x_center < depth_image.shape[1] and
                y_center < depth_image.shape[0]):
                depth = depth_image[y_center, x_center]
            else:
                depth = 0.0

            # Convert 2D pixel coordinates to 3D world coordinates
            if depth > 0:
                world_pos = self.pixel_to_world(
                    x_center, y_center, depth, self.intrinsic_matrix
                )

                det['position_3d'] = world_pos
                det['distance'] = depth
                detections_3d.append(det)

        return detections_3d

    def pixel_to_world(self, u, v, depth, intrinsic_matrix):
        """
        Convert pixel coordinates + depth to world coordinates
        """
        # Intrinsic matrix: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]

        # Convert to normalized coordinates
        x_norm = (u - cx) / fx
        y_norm = (v - cy) / fy

        # Convert to world coordinates
        world_x = x_norm * depth
        world_y = y_norm * depth
        world_z = depth

        return [world_x, world_y, world_z]

    def get_object_point_cloud(self, rgb_image, depth_image, detection):
        """
        Extract point cloud for a specific detected object
        """
        bbox = detection['bbox']
        x, y, w, h = [int(coord) for coord in bbox]

        # Ensure coordinates are within image bounds
        h, w_img, _ = rgb_image.shape if len(rgb_image.shape) == 3 else (rgb_image.shape[0], rgb_image.shape[1], 1)
        d_h, d_w = depth_image.shape

        x = max(0, min(x, d_w - 1))
        y = max(0, min(y, d_h - 1))
        w = min(w, d_w - x)
        h = min(h, d_h - y)

        # Extract region of interest
        roi_depth = depth_image[y:y+h, x:x+w]
        roi_rgb = rgb_image[y:y+h, x:x+w] if len(rgb_image.shape) == 3 else np.repeat(rgb_image[y:y+h, x:x+w][:,:,np.newaxis], 3, axis=2)

        # Create point cloud
        points = []
        colors = []

        for v in range(h):
            for u in range(w):
                depth_val = roi_depth[v, u]
                if depth_val > 0 and not np.isnan(depth_val):
                    world_pos = self.pixel_to_world(
                        x + u, y + v, depth_val, self.intrinsic_matrix
                    )
                    points.append(world_pos)
                    colors.append(roi_rgb[v, u] if len(roi_rgb.shape) == 3 else [roi_rgb[v, u]]*3)

        return np.array(points), np.array(colors)
```

## Object Tracking and Association

### Multi-Object Tracker:
```python
import copy

class ObjectTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.max_disappeared = 30  # frames
        self.max_distance = 50     # pixels

    def update(self, detections):
        """
        Update object tracks with new detections
        """
        # If no tracks exist, create tracks for all detections
        if len(self.tracks) == 0:
            for det in detections:
                self.register(det)
        else:
            # Calculate distance matrix between existing tracks and new detections
            track_ids = list(self.tracks.keys())
            track_centroids = [track['centroid'] for track in self.tracks.values()]

            D = np.linalg.norm(
                np.array(track_centroids)[:, np.newaxis] -
                np.array([det['center'] for det in detections])[np.newaxis, :],
                axis=2
            )

            # Find optimal assignment
            rows, cols = self.linear_assignment(D)

            # Update assigned tracks
            used_row_idxs = set()
            used_col_idxs = set()

            for (row, col) in zip(rows, cols):
                if D[row, col] > self.max_distance:
                    continue

                track_id = track_ids[row]
                self.tracks[track_id]['centroid'] = detections[col]['center']
                self.tracks[track_id]['bbox'] = detections[col]['bbox']
                self.tracks[track_id]['class_name'] = detections[col]['class_name']
                self.tracks[track_id]['confidence'] = detections[col]['confidence']
                self.tracks[track_id]['disappeared'] = 0

                used_row_idxs.add(row)
                used_col_idxs.add(col)

            # Handle unassigned tracks (mark as disappeared)
            unused_row_idxs = set(range(0, D.shape[0])).difference(used_row_idxs)
            for row in unused_row_idxs:
                track_id = track_ids[row]
                self.tracks[track_id]['disappeared'] += 1

                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    self.deregister(track_id)

            # Handle unassigned detections (create new tracks)
            unused_col_idxs = set(range(0, D.shape[1])).difference(used_col_idxs)
            for col in unused_col_idxs:
                self.register(detections[col])

        # Return active tracks
        return self.tracks

    def register(self, detection):
        """
        Register a new detection as a track
        """
        self.tracks[self.next_id] = {
            'centroid': detection['center'],
            'bbox': detection['bbox'],
            'class_name': detection['class_name'],
            'confidence': detection['confidence'],
            'disappeared': 0,
            'detection_history': [detection]
        }
        self.next_id += 1

    def deregister(self, track_id):
        """
        Deregister a track
        """
        del self.tracks[track_id]

    def linear_assignment(self, cost_matrix):
        """
        Simple greedy assignment (in practice, use Hungarian algorithm)
        """
        # For simplicity, using greedy assignment
        # In practice, use scipy.optimize.linear_sum_assignment
        rows, cols = [], []
        cost_copy = cost_matrix.copy()

        for _ in range(min(cost_copy.shape)):
            # Find minimum cost
            min_idx = np.unravel_index(np.argmin(cost_copy, axis=None), cost_copy.shape)
            rows.append(min_idx[0])
            cols.append(min_idx[1])

            # Set row and column to infinity to prevent reassignment
            cost_copy[min_idx[0], :] = np.inf
            cost_copy[:, min_idx[1]] = np.inf

        return rows, cols

class PersistentObjectMemory:
    def __init__(self):
        self.object_database = {}  # Persistent storage of objects
        self.tracked_objects = {}  # Currently tracked objects
        self.memory_expiry = 300   # seconds (5 minutes)

    def update_object_memory(self, detections, current_time):
        """
        Update persistent object memory with new detections
        """
        for det in detections:
            object_key = self.create_object_key(det)

            if object_key in self.object_database:
                # Update existing object
                self.update_existing_object(object_key, det, current_time)
            else:
                # Create new object entry
                self.create_new_object(object_key, det, current_time)

    def create_object_key(self, detection):
        """
        Create a unique key for an object based on its class and approximate location
        """
        class_name = detection['class_name']
        center = detection['center']

        # Quantize position to group nearby objects of same class
        quantized_x = int(center[0] / 100)  # Group every 100 pixels
        quantized_y = int(center[1] / 100)

        return f"{class_name}_{quantized_x}_{quantized_y}"

    def update_existing_object(self, key, detection, current_time):
        """
        Update an existing object in memory
        """
        obj = self.object_database[key]

        # Update position (with smoothing)
        alpha = 0.7  # smoothing factor
        obj['position'][0] = alpha * detection['center'][0] + (1 - alpha) * obj['position'][0]
        obj['position'][1] = alpha * detection['center'][1] + (1 - alpha) * obj['position'][1]

        # Update last seen time
        obj['last_seen'] = current_time

        # Update confidence
        obj['confidence'] = max(obj['confidence'], detection['confidence'])

    def create_new_object(self, key, detection, current_time):
        """
        Create a new object entry in memory
        """
        self.object_database[key] = {
            'class_name': detection['class_name'],
            'position': detection['center'].copy(),
            'bbox': detection['bbox'].copy(),
            'confidence': detection['confidence'],
            'first_seen': current_time,
            'last_seen': current_time,
            'appearance_features': self.extract_features(detection)
        }

    def get_known_objects(self, class_filter=None, time_threshold=60):
        """
        Get objects that have been seen recently
        """
        current_time = time.time()
        valid_objects = []

        for key, obj in self.object_database.items():
            if current_time - obj['last_seen'] <= time_threshold:
                if class_filter is None or obj['class_name'] == class_filter:
                    valid_objects.append(obj)

        return valid_objects

    def extract_features(self, detection):
        """
        Extract appearance features for object re-identification
        """
        # In practice, this would use deep features from CNN
        # For now, return simplified features
        return {
            'bbox_size': detection['bbox'][2] * detection['bbox'][3],
            'aspect_ratio': detection['bbox'][2] / detection['bbox'][3],
            'center_x': detection['center'][0],
            'center_y': detection['center'][1]
        }

    def cleanup_expired_objects(self, current_time):
        """
        Remove objects that haven't been seen for too long
        """
        expired_keys = []
        for key, obj in self.object_database.items():
            if current_time - obj['last_seen'] > self.memory_expiry:
                expired_keys.append(key)

        for key in expired_keys:
            del self.object_database[key]
```

## Integration with Manipulation System

### Object Detection for Manipulation:
```python
class ManipulationObjectDetector:
    def __init__(self, base_detector):
        self.detector = base_detector
        self.manipulation_requirements = {
            'minimum_size': 0.05,  # meters
            'maximum_distance': 2.0,  # meters
            'required_class_confidence': 0.7
        }

    def find_manipulable_objects(self, scene_image, preferred_objects=None):
        """
        Find objects that can be manipulated by the robot
        """
        detections = self.detector.detect_robotic_objects(scene_image)

        manipulable_objects = []
        for det in detections:
            if self.is_manipulable(det, scene_image):
                # Check if this is a preferred object type (if specified)
                if preferred_objects is None or det['class_name'] in preferred_objects:
                    manipulable_objects.append(det)

        return manipulable_objects

    def is_manipulable(self, detection, image):
        """
        Check if an object is manipulable based on various criteria
        """
        # Check confidence
        if detection['confidence'] < self.manipulation_requirements['required_class_confidence']:
            return False

        # Check size (approximate from 2D bounding box)
        bbox = detection['bbox']
        bbox_area = bbox[2] * bbox[3]  # width * height

        # Convert pixel area to approximate real-world size
        # This is a simplification - in practice, use depth information
        approx_size = np.sqrt(bbox_area) / 100  # Rough conversion

        if approx_size < self.manipulation_requirements['minimum_size']:
            return False

        # Check if object properties indicate it's manipulable
        properties = detection.get('properties', {})
        if not properties.get('manipulable', True):
            return False

        return True

    def select_best_object_for_task(self, available_objects, task_requirements):
        """
        Select the best object for a specific manipulation task
        """
        if not available_objects:
            return None

        # Score each object based on task requirements
        scored_objects = []
        for obj in available_objects:
            score = self.score_object_for_task(obj, task_requirements)
            scored_objects.append((obj, score))

        # Return object with highest score
        best_obj, best_score = max(scored_objects, key=lambda x: x[1])
        return best_obj

    def score_object_for_task(self, object_info, task_requirements):
        """
        Score an object based on how well it matches task requirements
        """
        score = 0.0

        # Task-specific scoring
        if 'object_type' in task_requirements:
            if object_info['class_name'] == task_requirements['object_type']:
                score += 10.0  # High score for exact match

        # Size-based scoring
        if 'size_preference' in task_requirements:
            size_pref = task_requirements['size_preference']
            bbox_area = object_info['bbox'][2] * object_info['bbox'][3]

            if size_pref == 'large' and bbox_area > 10000:
                score += 5.0
            elif size_pref == 'small' and bbox_area < 5000:
                score += 5.0

        # Confidence-based scoring
        score += object_info['confidence'] * 10

        # Proximity scoring (simplification)
        # In practice, consider actual distance using depth data
        score += 2.0  # Base score

        return score

    def generate_grasp_plan(self, object_info):
        """
        Generate a grasp plan for the detected object
        """
        # Get grasp points from object detection
        grasp_points = self.detector.get_object_grasp_points(object_info)

        # Create grasp plan
        grasp_plan = {
            'object_id': object_info['class_name'],
            'grasp_points': grasp_points,
            'preferred_approach': self.select_approach_direction(object_info),
            'grasp_type': self.select_grasp_type(object_info),
            'safety_margin': 0.05  # meters
        }

        return grasp_plan

    def select_approach_direction(self, object_info):
        """
        Select the best approach direction for grasping
        """
        # Analyze object shape and orientation to determine best approach
        bbox = object_info['bbox']

        # For now, choose approach direction based on object dimensions
        if bbox[2] > bbox[3]:  # width > height
            # Approach from top or bottom
            return 'top'
        else:
            # Approach from side
            return 'side'

    def select_grasp_type(self, object_info):
        """
        Select the appropriate grasp type for the object
        """
        class_name = object_info['class_name']
        properties = object_info.get('properties', {})

        if class_name in ['cup', 'mug']:
            return 'cylindrical'  # Grasp around the cylinder
        elif class_name in ['book', 'box']:
            return 'top'  # Grasp from top
        elif class_name in ['bottle']:
            return 'parallel'  # Parallel jaw grasp on neck
        else:
            return 'pinch'  # Default pinch grasp
```

## Performance Optimization

### Efficient Detection Pipeline:
```python
import threading
from queue import Queue
import time

class EfficientDetectionPipeline:
    def __init__(self, detector, num_threads=2):
        self.detector = detector
        self.input_queue = Queue(maxsize=10)
        self.output_queue = Queue(maxsize=10)
        self.num_threads = num_threads
        self.running = False
        self.threads = []

    def start_pipeline(self):
        """
        Start the detection pipeline with multiple threads
        """
        self.running = True

        # Start processing threads
        for i in range(self.num_threads):
            thread = threading.Thread(target=self.process_loop, args=(i,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

    def process_loop(self, thread_id):
        """
        Main processing loop for each thread
        """
        while self.running:
            try:
                # Get input from queue with timeout
                input_data = self.input_queue.get(timeout=1.0)

                # Process the data
                results = self.detector.detect_robotic_objects(input_data['image'])

                # Add results to output queue
                output_item = {
                    'results': results,
                    'timestamp': input_data['timestamp'],
                    'thread_id': thread_id
                }

                self.output_queue.put(output_item)

            except:
                # Timeout or other exception, continue loop
                continue

    def submit_image(self, image):
        """
        Submit an image for detection
        """
        input_item = {
            'image': image,
            'timestamp': time.time()
        }

        # Add to input queue if not full
        if not self.input_queue.full():
            self.input_queue.put(input_item)
            return True
        else:
            return False  # Queue full

    def get_results(self, timeout=0.1):
        """
        Get detection results from the pipeline
        """
        try:
            return self.output_queue.get(timeout=timeout)
        except:
            return None  # No results available

    def stop_pipeline(self):
        """
        Stop the detection pipeline
        """
        self.running = False

        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=2.0)

class DetectionResultAggregator:
    def __init__(self):
        self.recent_results = []
        self.max_results = 10
        self.temporal_window = 0.5  # seconds

    def add_results(self, detection_results):
        """
        Add new detection results to the aggregator
        """
        self.recent_results.append(detection_results)

        # Keep only recent results
        current_time = time.time()
        self.recent_results = [
            res for res in self.recent_results
            if current_time - res.get('timestamp', current_time) < self.temporal_window
        ]

    def get_stable_detections(self):
        """
        Get stable detections by combining multiple frames
        """
        if not self.recent_results:
            return []

        # Group detections by object class and position
        grouped_detections = {}
        for result in self.recent_results:
            for det in result.get('results', []):
                # Create a unique identifier for similar detections
                key = f"{det['class_name']}_{int(det['center'][0]/50)}_{int(det['center'][1]/50)}"

                if key not in grouped_detections:
                    grouped_detections[key] = {
                        'detections': [],
                        'class_name': det['class_name']
                    }

                grouped_detections[key]['detections'].append(det)

        # Aggregate grouped detections
        stable_detections = []
        for key, group in grouped_detections.items():
            if len(group['detections']) >= 2:  # Require detection in at least 2 frames
                # Average the positions and use max confidence
                avg_x = np.mean([det['center'][0] for det in group['detections']])
                avg_y = np.mean([det['center'][1] for det in group['detections']])
                max_conf = max(det['confidence'] for det in group['detections'])

                # Create stable detection
                stable_det = {
                    'class_name': group['class_name'],
                    'center': [avg_x, avg_y],
                    'confidence': max_conf,
                    'bbox': group['detections'][0]['bbox']  # Use first detection's bbox as reference
                }

                stable_detections.append(stable_det)

        return stable_detections
```

## Integration with Overall System

### Object Detection System Integration:
```python
class ObjectDetectionSystem:
    def __init__(self, camera_intrinsic_matrix):
        # Initialize detection components
        self.rgb_detector = RoboticsObjectDetector()
        self.rgbd_detector = RGBDObjectDetector(camera_intrinsic_matrix)
        self.tracker = ObjectTracker()
        self.memory = PersistentObjectMemory()
        self.manipulation_detector = ManipulationObjectDetector(self.rgb_detector)
        self.pipeline = EfficientDetectionPipeline(self.rgb_detector)
        self.aggregator = DetectionResultAggregator()

        # Start processing pipeline
        self.pipeline.start_pipeline()

    def process_camera_frame(self, rgb_image, depth_image=None):
        """
        Process a camera frame to detect and track objects
        """
        current_time = time.time()

        # Submit image to detection pipeline
        self.pipeline.submit_image(rgb_image)

        # Get results from pipeline
        pipeline_result = self.pipeline.get_results()
        if pipeline_result:
            # Add to aggregator
            self.aggregator.add_results(pipeline_result)

        # Get stable detections
        stable_detections = self.aggregator.get_stable_detections()

        # Update tracker
        tracked_objects = self.tracker.update(stable_detections)

        # Update persistent memory
        self.memory.update_object_memory(stable_detections, current_time)

        # If depth image available, compute 3D positions
        if depth_image is not None:
            detections_3d = self.rgbd_detector.detect_and_localize(rgb_image, depth_image)
            return detections_3d

        return stable_detections

    def get_objects_for_task(self, task_description, scene_image):
        """
        Get relevant objects for a specific task
        """
        # Find manipulable objects in scene
        manipulable_objects = self.manipulation_detector.find_manipulable_objects(
            scene_image,
            preferred_objects=self.extract_preferred_objects(task_description)
        )

        # Select best object for task
        best_object = self.manipulation_detector.select_best_object_for_task(
            manipulable_objects,
            self.parse_task_requirements(task_description)
        )

        return {
            'all_objects': manipulable_objects,
            'selected_object': best_object,
            'grasp_plan': self.manipulation_detector.generate_grasp_plan(best_object) if best_object else None
        }

    def extract_preferred_objects(self, task_description):
        """
        Extract preferred object types from task description
        """
        # Simple keyword matching (in practice, use NLP)
        preferred = []
        desc_lower = task_description.lower()

        if 'cup' in desc_lower or 'mug' in desc_lower or 'drink' in desc_lower:
            preferred.append('cup')
        if 'book' in desc_lower or 'read' in desc_lower:
            preferred.append('book')
        if 'bottle' in desc_lower or 'water' in desc_lower:
            preferred.append('bottle')
        if 'box' in desc_lower or 'container' in desc_lower:
            preferred.append('box')

        return preferred if preferred else None

    def parse_task_requirements(self, task_description):
        """
        Parse task requirements from description
        """
        requirements = {}

        # Simple parsing (in practice, use more sophisticated NLP)
        if 'large' in task_description.lower():
            requirements['size_preference'] = 'large'
        elif 'small' in task_description.lower():
            requirements['size_preference'] = 'small'
        elif 'big' in task_description.lower():
            requirements['size_preference'] = 'large'

        return requirements

    def cleanup(self):
        """
        Cleanup resources
        """
        self.pipeline.stop_pipeline()
```

The object detection integration component provides the Physical AI system with the ability to perceive and understand its environment by identifying and localizing objects. This capability is essential for both navigation (obstacle detection) and manipulation (object identification and grasp planning) tasks.