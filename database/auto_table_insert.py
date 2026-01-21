# Auto table creation and insertion
import sys
from datetime import datetime
from connect import get_connection

# Manages date-based table partitioning
class AutoTableInserter:
    
    def __init__(self):
        self.db = None
        self.created_tables = set()
        
    def connect(self):
        self.db = get_connection()
        if not self.db:
            print("Connection failed")
            return False
        
        all_tables = self.db.get_tables()
        for table_name in all_tables:
            if table_name.startswith('flir_data_'):
                self.created_tables.add(table_name)
        return True
    
    def disconnect(self):
        if self.db:
            self.db.disconnect()
            
    # Generate table name from timestamp
    def get_table_name(self, timestamp):
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        return f"flir_data_{timestamp.strftime('%Y%m%d')}"
    
    # Create table if needed
    def create_table_if_not_exists(self, table_name):
        if table_name in self.created_tables:
            return True
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME NOT NULL,
            data_id INT NOT NULL,
            x_axis FLOAT NOT NULL,
            y_axis FLOAT NOT NULL,
            valid BOOLEAN NOT NULL DEFAULT TRUE,
            INDEX idx_timestamp (timestamp),
            INDEX idx_data_id (data_id)
        )
        """
        self.db.cursor.execute(create_table_query)
        self.db.connection.commit()
        self.created_tables.add(table_name)
        return True
        
    # Insert coordinate data
    def insert_data(self, timestamp, data_id, x_axis, y_axis, valid=True):
        if not self.db:
            print("Connection not established")
            return False
        table_name = self.get_table_name(timestamp)
        if not self.create_table_if_not_exists(table_name):
            return False
        insert_query = f"INSERT INTO {table_name} (timestamp, data_id, x_axis, y_axis, valid) VALUES (%s, %s, %s, %s, %s)"
        return self.db.execute_update(insert_query, (timestamp, data_id, x_axis, y_axis, valid))
