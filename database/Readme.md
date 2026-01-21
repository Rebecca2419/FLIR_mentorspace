# Database Module Documentation

## Database Files

### connect.py
Database connection manager providing MySQL connectivity.

**Usage:**
```python
from connect import get_connection
db = get_connection()
results = db.execute_query("SELECT * FROM table")
db.disconnect()
```

### auto_table_insert.py
 Creates date-based tables (`flir_data_YYYYMMDD`) dynamically

**Usage:**
```python
from auto_table_insert import AutoTableInserter
inserter = AutoTableInserter()
inserter.connect()
inserter.insert_data(timestamp, data_id, x_axis, y_axis, valid=True)
inserter.disconnect()
```

## Data Import & Processing

### import_coordinates.py
Parses raw coordinate data from text files and imports into database.

**Usage:**
```bash
python import_coordinates.py <data_id> <input_file.txt> <YYYY-MM-DD>
```

### filter.py
filtering with statistical analysis (IQR, skewness), interpolation.

**Usage:**
```bash
python filter.py <input_file.txt> <duration_seconds> <YYYY-MM-DD>
```

### split_trajectories.py
Separates 2-person trajectories.

**Usage:**
```bash
python split_trajectories.py <input_file.txt> <output_A.txt> <output_B.txt>
```

## Stop Detection

### stop_detection_x_only.py
Detects stop points using X-axis velocity thresholding.

**Usage:**
```bash
python stop_detection_x_only.py <input_file.txt> <velocity_threshold> <cluster>
```

## Visualization Tools

### UI_grid.py

**Usage:**
```bash
python UI_grid.py [--timestamp YYYY-MM-DD_HH-MM-SS] [--date YYYY-MM-DD]
```

### UI_y_quantized.py
Y-axis quantized visualization mode.

**Usage:**
```bash
python UI_y_quantized.py [--timestamp YYYY-MM-DD_HH-MM-SS] [--date YYYY-MM-DD]
```

### show_stops_y_quantized.py
Visualizes stop locations on room layout.

**Usage:**
```bash
python show_stops_y_quantized.py <stops_file.txt> [additional_files...]
```
