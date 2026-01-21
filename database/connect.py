# Database connection module
import sys
import mysql.connector
from config import DB_CONFIG

# Connection wrapper for MySQL
class DatabaseConnection:
    def __init__(self, config=None):
        self.config = config or DB_CONFIG
        self.connection = None
        self.cursor = None
        
    # Establish connection
    def connect(self):
        self.connection = mysql.connector.connect(**self.config)
        self.cursor = self.connection.cursor(dictionary=True)
        print("Connection successful")
        return True
        
    def disconnect(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            
    # Run SELECT queries
    def execute_query(self, query, params=None):
        if not self.connection:
            print("Connection not established")
            return []
        self.cursor.execute(query, params or ())
        return self.cursor.fetchall()
        
    # Run INSERT/UPDATE queries
    def execute_update(self, query, params=None):
        if not self.connection:
            print("Connection not established")
            return False
        self.cursor.execute(query, params or ())
        self.connection.commit()
        return True
        
    def get_tables(self):
        query = "SELECT TABLE_NAME FROM information_schema.TABLES WHERE TABLE_SCHEMA = %s"
        results = self.execute_query(query, (self.config['database'],))
        return [row['TABLE_NAME'] for row in results]
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        
def get_connection(config=None):
    db = DatabaseConnection(config)
    if db.connect():
        return db
    return None
