import pandas as pd
import ibm_db
import configparser

# Read configuration from config.ini file
config = configparser.ConfigParser()
config.read('../config.ini')

# User Config
dsn_database = config['DataIngestion']["dsn_database"]
dsn_uid = config['DataIngestion']["dsn_uid"]
dsn_pwd = config['DataIngestion']["dsn_pwd"]
dsn_hostname = config['DataIngestion']["dsn_hostname"]
dsn_port = config['DataIngestion']["dsn_port"]
dsn_protocol = config['DataIngestion']["dsn_protocol"]
dsn_driver = config['DataIngestion']["dsn_driver"]
filename = config['DataIngestion']["filename"]
table_name = config['DataIngestion']["table_name"]


def db2_connector():
    dsn = ("DRIVER={{IBM DB2 ODBC DRIVER}};" "DATABASE={0};" "HOSTNAME={1};" "PORT={2};" "PROTOCOL=TCPIP;" "UID={3};" "PWD={4};SECURITY=SSL").format(dsn_database, dsn_hostname, dsn_port, dsn_uid, dsn_pwd)
    options = {ibm_db.SQL_ATTR_AUTOCOMMIT: ibm_db.SQL_AUTOCOMMIT_ON}
    conn = ibm_db.connect(dsn, "", "", options)
    return conn,ibm_db

def db2_connection_close(conn):
    ibm_db.close(conn)

