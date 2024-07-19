import pandas as pd
import ibm_db
import configparser

# # Read configuration from expertConfig.ini file
expertConfig = configparser.ConfigParser()
expertConfig.read('./../expertConfig.ini')

# super_config = configparser.ConfigParser()
# super_config.read('./../simpleConfig.ini')
# home_dir  = super_config['Default']['home_dir']


 
# User expertConfig
dsn_database = expertConfig['DataIngestion']["dsn_database"]
dsn_uid = expertConfig['DataIngestion']["dsn_uid"]
dsn_pwd = expertConfig['DataIngestion']["dsn_pwd"]
dsn_hostname = expertConfig['DataIngestion']["dsn_hostname"]
dsn_port = expertConfig['DataIngestion']["dsn_port"]
dsn_protocol = expertConfig['DataIngestion']["dsn_protocol"]
dsn_driver = expertConfig['DataIngestion']["dsn_driver"]


def db2_connector():
    dsn = ("DRIVER={{IBM DB2 ODBC DRIVER}};" "DATABASE={0};" "HOSTNAME={1};" "PORT={2};" "PROTOCOL=TCPIP;" "UID={3};" "PWD={4};SECURITY=SSL").format(dsn_database, dsn_hostname, dsn_port, dsn_uid, dsn_pwd)
    options = {ibm_db.SQL_ATTR_AUTOCOMMIT: ibm_db.SQL_AUTOCOMMIT_ON}
    conn = ibm_db.connect(dsn, "", "", options)
    return conn

def db2_connection_close(conn):
    ibm_db.close(conn)


def db2_connectorWithSchema(schema_name):
    schema_name =schema_name.upper()
    dsn = ("DRIVER={{IBM DB2 ODBC DRIVER}};"
           "DATABASE={0};"
           "HOSTNAME={1};"
           "PORT={2};"
           "PROTOCOL=TCPIP;"
           "UID={3};"
           "PWD={4};"
           "SECURITY=SSL;"
           "CURRENTSCHEMA={5}").format(dsn_database, dsn_hostname, dsn_port, dsn_uid, dsn_pwd, schema_name)

    options = {ibm_db.SQL_ATTR_AUTOCOMMIT: ibm_db.SQL_AUTOCOMMIT_ON}
    conn = ibm_db.connect(dsn, "", "", options)

    return conn
