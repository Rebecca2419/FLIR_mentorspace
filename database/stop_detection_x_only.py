

# Detect stops using X-axis velocity
import sys
import argparse
import numpy as np
from collections import defaultdict
from import_coordinates import parse_coordinates

# Transform to room space
def transform_coordinates(x, y):
    x_new = x / 1000.0 + 3.45
    y_new = -(y / 1000.0) + 4.15
    x_new = max(0, min(6.9, x_new))
    y_new = max(0, min(8.3, y_new))
    
    return (x_new, y_new)

# Identify stop clusters from velocity
def detect_stops_x_only(coordinates, velocity_threshold=1.8, min_cluster_size=45):
    x_velocities = []
    for i in range(1, len(coordinates)):
        dx = abs(coordinates[i][0] - coordinates[i-1][0])
        x_velocities.append(dx)
    stopped_frames = []
    for i, v in enumerate(x_velocities):
        if v < velocity_threshold:
            stopped_frames.append(i + 1)
    
    if not stopped_frames:
        return []
    
    stop_clusters = []
    current_cluster = [stopped_frames[0]]
    
    for frame in stopped_frames[1:]:
        if frame == current_cluster[-1] + 1:
            current_cluster.append(frame)
        else:
            if len(current_cluster) >= min_cluster_size:
                stop_clusters.append(current_cluster[:])
            current_cluster = [frame]
    
    if len(current_cluster) >= min_cluster_size:
        stop_clusters.append(current_cluster)
    
    merged_clusters = merge_clusters_x_only(stop_clusters, coordinates, max_gap=5, max_x_distance=0.2)

    stops = []
    for cluster in merged_clusters:
        cluster_points = [coordinates[i] for i in cluster]
        
        x_values = [p[0] for p in cluster_points]
        y_values = [p[1] for p in cluster_points]
        
        stop_info = {
            "x": np.mean(x_values),
            "y": np.mean(y_values)
        }
        stops.append(stop_info)
    
    return stops

# Merge nearby stop clusters
def merge_clusters_x_only(clusters, coordinates, max_gap=5, max_x_distance=0.2):
    merged = [clusters[0][:]]
    
    for current in clusters[1:]:
        last_merged = merged[-1]
        gap = current[0] - last_merged[-1]
        
        last_center_idx = np.mean(last_merged)
        current_center_idx = np.mean(current)
        
        last_center_pt = coordinates[int(last_center_idx)]
        current_center_pt = coordinates[int(current_center_idx)]
        
        x_dist = abs(current_center_pt[0] - last_center_pt[0])
        
        if gap <= max_gap and x_dist <= max_x_distance:
            merged[-1].extend(current)
        else:
            merged.append(current[:])
    
    return merged

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('velocity_threshold', type=float)
    parser.add_argument('min_cluster_size', type=int)
    
    args = parser.parse_args()
    
    # Read input file
    with open(args.input_file, 'r') as f:
        lines = f.readlines()
    
    # Parse coordinates
    coordinates = []
    last_raw_position = None
    
    for i, line in enumerate(lines):
        coord = parse_coordinates(line, last_raw_position)
        if coord is None:
            continue
        x_raw, y_raw = coord
        x_transformed, y_transformed = transform_coordinates(x_raw, y_raw)
        coordinates.append((x_transformed, y_transformed))
        last_raw_position = coord
    
    # Detect stops
    stops = detect_stops_x_only(
        coordinates,
        velocity_threshold=args.velocity_threshold,
        min_cluster_size=args.min_cluster_size
    )
    
    # Write output file
    output_file = args.input_file.rsplit('.', 1)[0] + '_stops_x_only.txt'
    with open(output_file, 'w') as f:
        f.write("ID,X,Y\n")
        for i, stop in enumerate(stops, 1):
            f.write(f"{i},{stop['x']:.4f},{stop['y']:.4f}\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
