from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class Detection:
    id: int
    centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    contour: List[Tuple[int, int]] = None  # optional
    possibility: float = None  # optional

@dataclass
class singleFrameResult:
    frame_id: int
    detections: List[Detection]

    def to_dict(self) -> Dict:
        return {
            "frame_id": self.frame_id,
            "detections": [
                {
                    "id": det.id,
                    "bbox": det.bbox,
                    "centroid": det.centroid,
                }
                for det in self.detections
            ],
        }
