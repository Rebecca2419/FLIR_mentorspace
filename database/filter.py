# Statistical filtering and interpolation
import sys
import argparse
import numpy as np
from scipy import stats
from scipy.stats import trim_mean
from datetime import datetime, timedelta
from config import FILTER_CONFIG
from auto_table_insert import AutoTableInserter
from import_coordinates import parse_coordinates
from restricted_zone import is_in_restricted_zone, find_outside_point
from interpolation import interpolate_points, find_anchor_point, apply_interpolation

# Transform to room space
def transform_coordinates(x, y):
    x_new = x / 1000.0 + 3.45
    y_new = -(y / 1000.0) + 4.15
    return (x_new, y_new)

# Select statistical method based on distribution
def get_strategy_value(data, iqr_val, skew_val):
    if iqr_val < 200:
        return np.median(data)
    elif abs(skew_val) < 0.6:
        return trim_mean(data, 0.1)
    elif skew_val < -0.6:
        return np.percentile(data, 90)
    else:
        return np.percentile(data, 10)

def meters_distance(a_raw, b_raw):
    ta = transform_coordinates(a_raw[0], a_raw[1])
    tb = transform_coordinates(b_raw[0], b_raw[1])
    dx = tb[0] - ta[0]
    dy = tb[1] - ta[1]
    return float((dx * dx + dy * dy) ** 0.5)

# Extract data bucket for downsampling
def get_bucket(index, bucket_size, valid_numbers, valid_index, data):
    start_idx = int(index * bucket_size)
    end_idx = min(int((index + 1) * bucket_size), len(data))
    bucket = []
    for j, line in enumerate(valid_index):
        if start_idx <= line < end_idx:
            bucket.append(valid_numbers[j])
    #Identify that if the bucket is empty
    empty = True
    for line in range(start_idx, end_idx):
        if line < 0 or line >= len(data):
            continue
        if data[line].strip():
            empty = False
            break
    return bucket, empty

# Insert with restricted zone validation
def insert_with_zone_check(inserter, timestamp, x_db, y_db, prev_written, valid=True):
    if is_in_restricted_zone(x_db, y_db):
        x_db, y_db = find_outside_point((x_db, y_db), prev_written)
    inserter.insert_data(timestamp, 1, x_db, y_db, valid=valid)
    return (x_db, y_db)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('time')
    parser.add_argument('date')
    
    args = parser.parse_args()
    
    inserter = AutoTableInserter()
    inserter.connect()
    
    base_date = datetime.strptime(args.date, '%Y-%m-%d')
    
    with open(args.input_file, 'r') as f:
        data = f.readlines()
    
    #Traverse the data to determine valid data.
    valid_numbers = []
    valid_index = []
    last_raw_position = None
    
    for i, line in enumerate(data):
        #Parse the coordinates of each line. If there is no value, return None.
        if line.strip() == "[]":
            continue
        coordinate = parse_coordinates(line, last_raw_position)
        if coordinate is None or coordinate == (0, 0):
            continue
        valid_numbers.append((float(coordinate[0]),float(coordinate[1])))
        valid_index.append(i)
        last_raw_position = coordinate
    
    #Determine the bucket size based on frequency requirement
    output_points = int(FILTER_CONFIG["sample_rate"] * float(args.time))
    bucket_size = len(data) / output_points
    
    # State tracking
    prev_pt_raw = None
    prev_m_written = None
    
    # Process each output point
    i = 0
    while i < output_points:
        bucket, empty = get_bucket(i, bucket_size, valid_numbers, valid_index, data)
        
        # Handle empty bucket
        if empty:
            timestamp = base_date + timedelta(seconds=i)
            inserter.insert_data(timestamp, 1, 0.0, 0.0, valid=False)
            prev_pt_raw = None
            i += 1
            continue
        
        # Extract coordinates
        x_coords = []
        y_coords = []
        for coord in bucket:
            x_coords.append(coord[0])
            y_coords.append(coord[1])
        
        # Handle no data
        if len(x_coords) == 0:
            timestamp = base_date + timedelta(seconds=i)
            inserter.insert_data(timestamp, 1, 0.0, 0.0, valid=False)
            i += 1
            continue
        
        # Use average for small samples
        if len(x_coords) < 3:
            avg_x = np.mean(x_coords)
            avg_y = np.mean(y_coords)
            x_db, y_db = transform_coordinates(avg_x, avg_y)
            timestamp = base_date + timedelta(seconds=i)
            inserter.insert_data(timestamp, 1, x_db, y_db, valid=True)
            i += 1
            continue
        
        # Calculate statistics
        x_arr = np.array(x_coords)
        x_iqr = stats.iqr(x_arr)
        x_skew = stats.skew(x_arr)
        
        y_arr = np.array(y_coords)
        y_iqr = stats.iqr(y_arr)
        y_skew = stats.skew(y_arr)
        
        # Select value using strategy
        x_value = get_strategy_value(x_arr, x_iqr, x_skew)
        y_value = get_strategy_value(y_arr, y_iqr, y_skew)
        proposed_pt_raw = (float(x_value), float(y_value))
        proposed_pt_m = transform_coordinates(proposed_pt_raw[0], proposed_pt_raw[1])

        # Handle first point
        if prev_pt_raw is None:
            prev_pt_raw = proposed_pt_raw
            timestamp = base_date + timedelta(seconds=i)
            prev_m_written = insert_with_zone_check(inserter, timestamp, proposed_pt_m[0], proposed_pt_m[1], None)
            i += 1
            continue

        # Check distance threshold
        dist_m = meters_distance(prev_pt_raw, proposed_pt_raw)
        if dist_m <= 2.0:
            prev_pt_raw = proposed_pt_raw
            timestamp = base_date + timedelta(seconds=i)
            prev_m_written = insert_with_zone_check(inserter, timestamp, proposed_pt_m[0], proposed_pt_m[1], prev_m_written)
            i += 1
            continue

        # Find anchor point for interpolation
        anchor_raw, anchor_index = find_anchor_point(
            i, output_points, prev_pt_raw,
            bucket_size, valid_numbers, valid_index, data,
            meters_distance
        )

        # No anchor found, use proposed point
        if anchor_raw is None:
            prev_pt_raw = proposed_pt_raw
            x_db, y_db = transform_coordinates(prev_pt_raw[0], prev_pt_raw[1])
            timestamp = base_date + timedelta(seconds=i)
            prev_m_written = insert_with_zone_check(inserter, timestamp, x_db, y_db, prev_m_written)
            i += 1
            continue

        # Apply interpolation between prev and anchor
        prev_m = transform_coordinates(prev_pt_raw[0], prev_pt_raw[1])
        anchor_m = transform_coordinates(anchor_raw[0], anchor_raw[1])
        
        prev_m_written = apply_interpolation(
            prev_m, anchor_m,
            i, anchor_index, base_date, inserter, prev_m_written
        )
        
        # Insert anchor point
        tm_anchor = base_date + timedelta(seconds=anchor_index)
        xd_anchor, yd_anchor = anchor_m
        prev_m_written = insert_with_zone_check(inserter, tm_anchor, xd_anchor, yd_anchor, prev_m_written)
        prev_pt_raw = anchor_raw
        i = anchor_index + 1
        continue
    
    inserter.disconnect()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
