# Split 2-person trajectories
import sys
import argparse
import math
import ast

# Parse coordinate line
def load_line(line):
    stripped = line.strip()
    if not stripped:
        return []
    data = ast.literal_eval(stripped)
    coords = []
    for item in data:
        x, y = item
        coords.append((float(x), float(y)))
    return coords

# Calculate distance between points
def dist(a, b):
    if a is None:
        return math.inf
    return math.hypot(a[0] - b[0], a[1] - b[1])

# Format coordinate for output
def format_coord(coord):
    if coord is None:
        return "[]"
    x, y = coord
    if x.is_integer():
        x = int(x)
    if y.is_integer():
        y = int(y)
    return f"[({x}, {y})]"

# Main splitting logic using minimum distance
def split_file(input_path, out_a, out_b):
    track_a = None
    track_b = None
    output_a = []
    output_b = []
    
    with open(input_path, 'r') as f:
        for line in f:
            coords = load_line(line)
            
            if not coords:
                output_a.append("[]\n")
                output_b.append("[]\n")
                continue
            
            if len(coords) == 1:
                c = coords[0]
                dist_a = dist(track_a, c)
                dist_b = dist(track_b, c)
                if dist_a <= dist_b:
                    track_a = c
                    output_a.append(format_coord(c) + "\n")
                    output_b.append("[]\n")
                else:
                    track_b = c
                    output_b.append(format_coord(c) + "\n")
                    output_a.append("[]\n")
                continue
            
            c1, c2 = coords[0], coords[1]
            option1 = dist(track_a, c1) + dist(track_b, c2)
            option2 = dist(track_a, c2) + dist(track_b, c1)
            
            if option2 < option1:
                c1, c2 = c2, c1
            
            track_a, track_b = c1, c2
            output_a.append(format_coord(c1) + "\n")
            output_b.append(format_coord(c2) + "\n")
    
    with open(out_a, 'w') as fa, open(out_b, 'w') as fb:
        fa.writelines(output_a)
        fb.writelines(output_b)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output_a')
    parser.add_argument('output_b')
    
    args = parser.parse_args()
    split_file(args.input, args.output_a, args.output_b)

if __name__ == "__main__":
    sys.exit(main())
