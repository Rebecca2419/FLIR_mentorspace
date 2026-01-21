def interpolate_points(prev_m, next_m, steps):
    if steps <= 0:
        return []
    dx = next_m[0] - prev_m[0]
    dy = next_m[1] - prev_m[1]
    return [(prev_m[0] + t * dx, prev_m[1] + t * dy) 
            for t in (k / float(steps) for k in range(1, steps + 1))]

def find_anchor_point(current_index, output_points, prev_pt_raw,
                     bucket_size, valid_numbers, valid_index, data,
                     distance_func):
    for j in range(current_index + 1, output_points):
        from filter import get_bucket
        cand_list, empty_j = get_bucket(j, bucket_size, valid_numbers, valid_index, data)
        if empty_j or not cand_list:
            continue
        
        best = min(cand_list, key=lambda c: distance_func(prev_pt_raw, c))
        best_dist = distance_func(prev_pt_raw, best)
        
        step_ahead = j - current_index
        threshold_m = 4.0 + 2.0 * (step_ahead - 1)
        
        if best_dist <= threshold_m:
            return best, j
    
    return None, None

from restricted_zone import is_in_restricted_zone, find_outside_point

def apply_interpolation(prev_m, anchor_m,
                       current_index, anchor_index, base_date,
                       inserter, last_written_m):
    steps = anchor_index - current_index
    filled = interpolate_points(prev_m, anchor_m, steps)
    for k in range(max(0, steps - 1)):
        idx = current_index + k
        from datetime import timedelta
        tm = base_date + timedelta(seconds=idx)
        xd, yd = filled[k]
        if is_in_restricted_zone(xd, yd):
            xd, yd = find_outside_point((xd, yd), last_written_m)
        inserter.insert_data(tm, 1, xd, yd)
        last_written_m = (xd, yd)
    
    return last_written_m
