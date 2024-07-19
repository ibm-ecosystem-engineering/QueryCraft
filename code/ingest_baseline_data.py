import sqlite3
import json

def create_db_and_table(db_name):
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(f"{db_name}.db")
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS contract (
            id TEXT ,
            supplier TEXT,
            services TEXT,
            effective_date TEXT,
            keywords TEXT,
            document_type TEXT,
            expiration_date TEXT,
            created_by TEXT,
            region TEXT,
            tcv REAL,
            term_type TEXT,
            title TEXT,
            keyword TEXT,
            status TEXT,
            functions TEXT,
            countries TEXT,
            regions TEXT
        )
    ''')
    
    # Commit changes and close connection
    conn.commit()
    conn.close()

def insert_data(db_name, data):
    # Connect to SQLite database
    conn = sqlite3.connect(f"{db_name}.db")
    cursor = conn.cursor()
    
    # Prepare data tuple from JSON object
    data_tuple = (
        data["id"], data["function"], data["supplier"], data["services"], 
        data["effective_date"], data["keywords"], data["document_type"], 
        data["expiration_date"], data["created_by"], data["region"], data["tcv"], 
        data["term_type"], data["title"], data["keyword"], data["status"], 
        data["functions"], data["countries"], data["regions"]
    )
    
    # Insert data into the table
    cursor.execute('''
        INSERT INTO contract(
            id, functions, supplier, services, effective_date, keywords, 
            document_type, expiration_date, created_by, region, tcv, 
            term_type, title, keyword, status, functions, countries, regions
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', data_tuple)
    
    # Commit changes and close connection
    conn.commit()
    conn.close()


if __name__ =="__main__":
    db_name = 'contracts_database'
    create_db_and_table(db_name)
    with open("./Baseline_data.json","r") as file:
        data = json.load(file)
        for obj in data:
            insert_data(db_name, obj)
    import csv
    with open("./Basline_dataset.csv", mode='w', newline='') as file:
        # Create a writer object from csv module
        csv_writer = csv.writer(file)
        
        # Add column headers
        csv_writer.writerow(data[0].keys())
        
        # Add rows
        for row in data:
            csv_writer.writerow(row.values())
