#util classes and functions 

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchreid
import torchvision.transforms as transforms
from ultralytics import YOLO
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PlayerIdentity:
    global_id: int
    embedding_gallery: deque = field(default_factory=lambda: deque(maxlen=50))
    color_histogram: Optional[np.ndarray] = None
    last_bbox: Optional[np.ndarray] = None
    last_seen_frame: int = 0
    total_appearances: int = 0
    team_id: int = -1  # -1 = unknown
    exit_edge: Optional[str] = None  # Where player left the frame
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    
    def get_mean_embedding(self) -> Optional[np.ndarray]:
        if len(self.embedding_gallery) == 0:
            return None
        return np.mean(list(self.embedding_gallery), axis=0)
    
    def get_best_similarity(self, query_emb: np.ndarray) -> float:
        if len(self.embedding_gallery) == 0:
            return 0.0
        similarities = [1 - cosine(query_emb, emb) for emb in self.embedding_gallery]
        return max(similarities)
    
    def get_weighted_similarity(self, query_emb: np.ndarray) -> float:
        if len(self.embedding_gallery) == 0:
            return 0.0
        
        embeddings = list(self.embedding_gallery)
        n = len(embeddings)
        weights = np.exp(np.linspace(-1, 0, n))
        weights /= weights.sum()
        
        similarities = [1 - cosine(query_emb, emb) for emb in embeddings]
        return np.average(similarities, weights=weights)
    
    def add_embedding(self, emb: np.ndarray, force: bool = False):
        if len(self.embedding_gallery) == 0 or force:
            self.embedding_gallery.append(emb.copy())
            return
        
        mean_emb = self.get_mean_embedding()
        similarity = 1 - cosine(emb, mean_emb)
        
        if 0.75 < similarity < 0.95:
            self.embedding_gallery.append(emb.copy())
        elif len(self.embedding_gallery) < 10:
            # Still building initial gallery
            self.embedding_gallery.append(emb.copy())


class EnhancedReIDManager:
    def __init__(
        self,
        similarity_threshold: float = 0.60,
        memory_seconds: float = 30.0,
        fps: int = 30,
        use_osnet: bool = True
    ):
        self.similarity_threshold = similarity_threshold
        self.memory_frames = int(memory_seconds * fps)
        self.fps = fps
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f" Using device: {self.device}")
        
        if use_osnet:
            print("Loading OSNet (ReID-optimized)...")
            self.feature_extractor = self._build_osnet()
        else:
            print("Loading ResNet18...")
            self.feature_extractor = self._build_resnet()
        
        self.active_players: Dict[int, PlayerIdentity] = {}  
        self.lost_players: Dict[int, PlayerIdentity] = {}    
        self.tracker_to_global: Dict[int, int] = {}          
        
        self.next_global_id = 1
        self.current_frame = 0
        
        self.frame_width = None
        self.frame_height = None
        self.edge_margin = 50
        
        self.team_embeddings: Dict[int, List[np.ndarray]] = {0: [], 1: []}
        
        self.stats = {
            'reidentifications': 0,
            'new_players': 0,
            'id_switches_prevented': 0
        }
    
    def _build_osnet(self):
        model = torchreid.models.build_model(
            name='osnet_ain_x1_0',
            num_classes=1000,
            loss='softmax',
            pretrained=True
        )
        model.eval()
        model.to(self.device)
        return model
    
    def _build_resnet(self):
        from torchvision.models import resnet18, ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
        model.to(self.device)
        return model
    
    def _extract_features(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(self.frame_width, x2) if self.frame_width else x2
        y2 = min(self.frame_height, y2) if self.frame_height else y2
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        try:
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (128, 256))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            crop = crop.astype(np.float32) / 255.0
            crop = (crop - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            crop = torch.from_numpy(crop).permute(2, 0, 1).float().unsqueeze(0)
            crop = crop.to(self.device)
            
            with torch.no_grad():
                features = self.feature_extractor(crop)
                features = F.normalize(features, p=2, dim=1)
            
            return features.cpu().numpy().flatten()
        except Exception as e:
            return None
    
    def _extract_color_histogram(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(self.frame_width, x2) if self.frame_width else x2
        y2 = min(self.frame_height, y2) if self.frame_height else y2
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        try:
            crop = frame[y1:y2, x1:x2]
            h = crop.shape[0]
            
            upper_body = crop[int(h * 0.15):int(h * 0.50), :]
            if upper_body.size == 0:
                return None
            
            hsv = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return hist
        except:
            return None
    
    def _get_edge_location(self, bbox: np.ndarray) -> Optional[str]:
        if self.frame_width is None:
            return None
        
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        
        edges = []
        if cx < self.edge_margin:
            edges.append('left')
        if cx > self.frame_width - self.edge_margin:
            edges.append('right')
        if cy < self.edge_margin:
            edges.append('top')
        if cy > self.frame_height - self.edge_margin:
            edges.append('bottom')
        
        return edges[0] if edges else None
    
    def _get_center(self, bbox: np.ndarray) -> np.ndarray:
        return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    
    def _compute_matching_score(
        self,
        query_emb: np.ndarray,
        query_hist: Optional[np.ndarray],
        query_bbox: np.ndarray,
        query_edge: Optional[str],
        candidate: PlayerIdentity
    ) -> Tuple[float, Dict]:

        breakdown = {}
        
        appearance_score = candidate.get_weighted_similarity(query_emb)
        breakdown['appearance'] = appearance_score
        
        color_score = 0.0
        if query_hist is not None and candidate.color_histogram is not None:
            color_score = cv2.compareHist(query_hist, candidate.color_histogram, cv2.HISTCMP_CORREL)
            color_score = max(0, color_score) 
        breakdown['color'] = color_score
        
        edge_bonus = 0.0
        if query_edge and candidate.exit_edge == query_edge:
            edge_bonus = 0.10  
        breakdown['edge_bonus'] = edge_bonus
        
        spatial_bonus = 0.0
        frames_gone = self.current_frame - candidate.last_seen_frame
        if candidate.last_bbox is not None and frames_gone < self.fps * 2: 
            query_center = self._get_center(query_bbox)
            last_center = self._get_center(candidate.last_bbox)
            
            predicted_center = last_center + candidate.velocity * frames_gone
            distance = np.linalg.norm(query_center - predicted_center)
            
            max_dist = max(self.frame_width, self.frame_height) * 0.5
            spatial_score = max(0, 1 - distance / max_dist)
            spatial_bonus = spatial_score * 0.15
        breakdown['spatial_bonus'] = spatial_bonus
        
        recency_factor = np.exp(-frames_gone / (self.fps * 10))  
        recency_bonus = recency_factor * 0.05
        breakdown['recency_bonus'] = recency_bonus
        
        final_score = (
            0.60 * appearance_score +
            0.15 * color_score +
            edge_bonus +
            spatial_bonus +
            recency_bonus
        )
        breakdown['final'] = final_score
        
        return final_score, breakdown
    
    def _find_best_matches(
        self,
        unmatched_detections: List[Tuple[int, np.ndarray, np.ndarray, Optional[np.ndarray]]]
    ) -> Dict[int, int]:
        if len(unmatched_detections) == 0 or len(self.lost_players) == 0:
            return {}
        
        # Build cost matrix
        lost_ids = list(self.lost_players.keys())
        cost_matrix = np.zeros((len(unmatched_detections), len(lost_ids)))
        
        for i, (det_idx, bbox, emb, hist) in enumerate(unmatched_detections):
            query_edge = self._get_edge_location(bbox)
            
            for j, global_id in enumerate(lost_ids):
                candidate = self.lost_players[global_id]
                
                if self.current_frame - candidate.last_seen_frame > self.memory_frames:
                    cost_matrix[i, j] = 999
                    continue
                
                score, _ = self._compute_matching_score(emb, hist, bbox, query_edge, candidate)
                cost_matrix[i, j] = 1 - score  
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = {}
        for r, c in zip(row_ind, col_ind):
            score = 1 - cost_matrix[r, c]
            if score > self.similarity_threshold:
                det_idx = unmatched_detections[r][0]
                global_id = lost_ids[c]
                matches[det_idx] = global_id
        
        return matches
    
    def _assign_team(self, embedding: np.ndarray, histogram: Optional[np.ndarray]) -> int:
        if len(self.team_embeddings[0]) < 3 or len(self.team_embeddings[1]) < 3:
            return -1  
        
        team0_centroid = np.mean(self.team_embeddings[0][-20:], axis=0)
        team1_centroid = np.mean(self.team_embeddings[1][-20:], axis=0)
        
        sim0 = 1 - cosine(embedding, team0_centroid)
        sim1 = 1 - cosine(embedding, team1_centroid)
        
        if abs(sim0 - sim1) > 0.1:  
            return 0 if sim0 > sim1 else 1
        return -1
    
    def update(
        self,
        frame: np.ndarray,
        detections: List[Tuple[int, np.ndarray]]
    ) -> List[Tuple[int, np.ndarray, int]]:
        self.current_frame += 1
        
        if self.frame_width is None:
            self.frame_height, self.frame_width = frame.shape[:2]
        
        current_tracker_ids: Set[int] = set()
        results = []
        unmatched_detections = []

        for tracker_id, bbox in detections:
            current_tracker_ids.add(tracker_id)
            
            embedding = self._extract_features(frame, bbox)
            histogram = self._extract_color_histogram(frame, bbox)
            
            if embedding is None:
                continue
            
            if tracker_id in self.tracker_to_global:
                global_id = self.tracker_to_global[tracker_id]
                player = self.active_players[global_id]
                
                if player.last_bbox is not None:
                    new_center = self._get_center(bbox)
                    old_center = self._get_center(player.last_bbox)
                    player.velocity = 0.7 * player.velocity + 0.3 * (new_center - old_center)
                
                player.add_embedding(embedding)
                if histogram is not None:
                    if player.color_histogram is None:
                        player.color_histogram = histogram
                    else:
                        player.color_histogram = 0.9 * player.color_histogram + 0.1 * histogram
                
                player.last_bbox = bbox.copy()
                player.last_seen_frame = self.current_frame
                player.total_appearances += 1
                
                results.append((global_id, bbox, player.team_id))
            else:
                unmatched_detections.append((tracker_id, bbox, embedding, histogram))

        if unmatched_detections:
            match_input = [
                (i, det[1], det[2], det[3]) 
                for i, det in enumerate(unmatched_detections)
            ]
            matches = self._find_best_matches(match_input)
            
            for i, (tracker_id, bbox, embedding, histogram) in enumerate(unmatched_detections):
                if i in matches:
                    global_id = matches[i]
                    player = self.lost_players.pop(global_id)
                    
                    player.add_embedding(embedding, force=True)
                    if histogram is not None:
                        player.color_histogram = histogram
                    player.last_bbox = bbox.copy()
                    player.last_seen_frame = self.current_frame
                    player.total_appearances += 1
                    
                    self.active_players[global_id] = player
                    self.tracker_to_global[tracker_id] = global_id
                    
                    self.stats['reidentifications'] += 1
                    print(f"Frame {self.current_frame}: Player {global_id} re-identified!")
                    
                    results.append((global_id, bbox, player.team_id))
                else:
                    global_id = self.next_global_id
                    self.next_global_id += 1
                    
                    player = PlayerIdentity(global_id=global_id)
                    player.add_embedding(embedding, force=True)
                    player.color_histogram = histogram
                    player.last_bbox = bbox.copy()
                    player.last_seen_frame = self.current_frame
                    player.total_appearances = 1
                    
                    player.team_id = self._assign_team(embedding, histogram)
                    
                    self.active_players[global_id] = player
                    self.tracker_to_global[tracker_id] = global_id
                    
                    self.stats['new_players'] += 1
                    print(f"Frame {self.current_frame}: New player {global_id}")
                    
                    results.append((global_id, bbox, player.team_id))
        
        disappeared_tracker_ids = set(self.tracker_to_global.keys()) - current_tracker_ids
        
        for tracker_id in disappeared_tracker_ids:
            global_id = self.tracker_to_global.pop(tracker_id)
            
            if global_id in self.active_players:
                player = self.active_players.pop(global_id)
                
                if player.last_bbox is not None:
                    player.exit_edge = self._get_edge_location(player.last_bbox)
                
                self.lost_players[global_id] = player
                print(f"Frame {self.current_frame}: Player {global_id} left (edge: {player.exit_edge})")
        
        to_remove = [
            gid for gid, player in self.lost_players.items()
            if self.current_frame - player.last_seen_frame > self.memory_frames
        ]
        for gid in to_remove:
            del self.lost_players[gid]
            print(f"Frame {self.current_frame}: Removed player {gid} from memory")
        
        return results
    
    def get_stats(self) -> Dict:
        return {
            'current_frame': self.current_frame,
            'active_players': len(self.active_players),
            'lost_players': len(self.lost_players),
            'total_players': self.next_global_id - 1,
            'reidentifications': self.stats['reidentifications'],
            'new_detections': self.stats['new_players']
        }
