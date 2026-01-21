
RESTRICTED_ZONES = [
    (1.55, 5.15, 2.12, 3.32),
    (1.55, 5.15, 4.58, 5.78),
]

def is_in_restricted_zone(x_m, y_m):
    return any(x_min <= x_m <= x_max and y_min <= y_m <= y_max 
               for x_min, x_max, y_min, y_max in RESTRICTED_ZONES)

def find_outside_point(point_m, prev_m):
    if prev_m is None:
        return point_m
    
    x, y = point_m
    candidates = []
    
    for x_min, x_max, y_min, y_max in RESTRICTED_ZONES:
        if x_min <= x <= x_max and y_min <= y <= y_max:
            candidates.extend([
                (x, y_min - 0.01), (x, y_max + 0.01),
                (x_min - 0.01, y), (x_max + 0.01, y)
            ])
    
    return min(candidates, key=lambda c: (c[0] - prev_m[0])**2 + (c[1] - prev_m[1])**2) if candidates else point_m
