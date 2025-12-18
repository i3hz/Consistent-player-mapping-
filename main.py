import cv2
import numpy as np
import torchvision.transforms as transforms
from ultralytics import YOLO
from typing import List
import warnings
from utils import *
warnings.filterwarnings('ignore')

class SoccerPlayerTracker:
    def __init__(
        self,
        model_path: str,
        input_video: str,
        output_video: str = 'output.mp4',
        similarity_threshold: float = 0.60,
        memory_seconds: float = 30.0
    ):
        self.model = YOLO(model_path)
        self.input_video = input_video
        self.output_video = output_video
        
        # Setup video
        self.cap = cv2.VideoCapture(input_video)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        
        self.writer = cv2.VideoWriter(
            output_video,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (self.width, self.height)
        )
        
        # Initialize ReID manager
        self.reid_manager = EnhancedReIDManager(
            similarity_threshold=similarity_threshold,
            memory_seconds=memory_seconds,
            fps=self.fps,
            use_osnet=True
        )
        
        # Visualization colors
        np.random.seed(42)
        self.colors = {
            i: tuple(map(int, np.random.randint(50, 255, 3)))
            for i in range(200)
        }
        
        # Team colors
        self.team_colors = {
            0: (255, 0, 0),    # Blue
            1: (0, 0, 255),    # Red
            -1: (0, 255, 0)    # Unknown - Green
        }
        
        print(f"Initialized tracker for: {input_video}")
        print(f"Resolution: {self.width}x{self.height} @ {self.fps}fps")
    
    def run(self, show_team_colors: bool = True, player_classes: List[int] = [1, 2]):
        print("\n" + "="*60)
        print("Starting tracking...")
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            results = self.model.track(
                frame,
                persist=True,
                classes=player_classes,
                verbose=False
            )
            
            detections = []
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                tracker_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for tracker_id, bbox, conf in zip(tracker_ids, boxes, confidences):
                    if conf > 0.3:  
                        detections.append((tracker_id, bbox))
            
            reid_results = self.reid_manager.update(frame, detections)
            
            annotated = frame.copy()
            
            for global_id, bbox, team_id in reid_results:
                x1, y1, x2, y2 = map(int, bbox)
                
                if show_team_colors:
                    color = self.team_colors.get(team_id, (0, 255, 0))
                else:
                    color = self.colors.get(global_id % 200, (0, 255, 0))
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                label = f"ID:{global_id}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - 25), (x1 + w + 10, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 5, y1 - 7),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            stats = self.reid_manager.get_stats()
            info_lines = [
                f"Frame: {frame_count}",
                f"Active: {stats['active_players']} | Memory: {stats['lost_players']}",
                f"Re-IDs: {stats['reidentifications']}"
            ]
            for i, line in enumerate(info_lines):
                cv2.putText(annotated, line, (10, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            self.writer.write(annotated)
            frame_count += 1
            
            if frame_count % 50 == 0:
                print(f"   Frame {frame_count} | Active: {stats['active_players']} | "
                      f"Memory: {stats['lost_players']} | Re-IDs: {stats['reidentifications']}")
        
        self.cap.release()
        self.writer.release()
        
        final_stats = self.reid_manager.get_stats()
        print(f"Total frames processed: {frame_count}")
        print(f"Total unique players: {final_stats['total_players']}")
        print(f"Total re-identifications: {final_stats['reidentifications']}")
        print(f"Output saved to: {self.output_video}")



tracker = SoccerPlayerTracker(
    model_path="best.pt",  
    input_video="15.mp4",
    output_video="output1.mp4",
    similarity_threshold=0.55,  
    memory_seconds=30.0
)

tracker.run(
    show_team_colors=False,
    player_classes=[1, 2]  
)
    
    