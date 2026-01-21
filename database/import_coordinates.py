# Import raw coordinates to database
import sys
import re
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import atexit
from auto_table_insert import AutoTableInserter

# Parse coordinate string
def parse_coordinates(line, last_position=None):
    line = line.strip()
    if line == "[]":
        return (0, 0)
    coords = re.findall(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", line)
    if not coords:
        return (0, 0)
    coords = [(int(x), int(y)) for x, y in coords]
    if not last_position:
        return coords[0]
    best = min(coords, key=lambda c: (c[0] - last_position[0])**2 + (c[1] - last_position[1])**2)
    return best

# Convert raw to room coordinates
def transform_coordinates(x, y):
    x_new = x / 1000.0 + 3.45
    y_new = -(y / 1000.0) + 4.15
    x_new = max(0, min(6.9, x_new))
    y_new = max(0, min(8.3, y_new))
    return (x_new, y_new)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int)
    parser.add_argument('txt_file')
    parser.add_argument('date')
    
    args = parser.parse_args()

    # Setup base timestamp
    base_date = datetime.strptime(args.date, '%Y-%m-%d')
    
    # Read input file
    path = Path(args.txt_file)
    lines = path.read_text().splitlines()
    
    # Connect to database
    inserter = AutoTableInserter()
    inserter.connect()

    atexit.register(inserter.disconnect)

    all_data = []
    valid_count = 0
    last_position = None

    # Parse all coordinates
    for line_index, line in enumerate(lines):
        coord = parse_coordinates(line, last_position)
        last_position = coord
        x_raw, y_raw = coord
        transformed = transform_coordinates(x_raw, y_raw)
        all_data.append((transformed[0], transformed[1], True))
        valid_count += 1
        print(f"{line_index + 1}: raw ({x_raw}, {y_raw}) -> transformed ({transformed[0]:.2f}, {transformed[1]:.2f}), valid=True")

    print(f"\nParsed completed: total {len(all_data)} lines, valid {valid_count}")

    # Insert into database
    inserted_count = 0
    for i, (x, y, valid) in enumerate(all_data):
        timestamp = base_date + timedelta(seconds=i)
        data_id = 1
        inserter.insert_data(timestamp, data_id, x, y, valid)
        status = "valid" if valid else "empty"
        print(f"{i+1}: ({x:.2f}, {y:.2f}) valid={valid} [{status}] time {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        inserted_count += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"- Total lines: {len(lines)}")
    print(f"- Inserted: {inserted_count}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
