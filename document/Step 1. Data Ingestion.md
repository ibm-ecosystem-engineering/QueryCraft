# <a name="_toc1471206741"></a>Step 1. Data Ingestion

Users have the flexibility to ingest data to both DB2 and SQLite databases, enhancing adaptability and ease of integration. Steps for both the options are explained below.

## <a name="_toc212047022"></a>Option 1.1. Delimited file Ingestion

You can ingest your delimited files to DB2 on IBM cloud. Once the data is properly ingested, you can run the QueryCraft pipeline on your data. 

#### *Prerequisites (First time activity):* 
- Access to IBM Cloud. You can create a free account. <https://cloud.ibm.com/registration>
- Access to a DB2 database. You can provision a free instance: <https://cloud.ibm.com/catalog/services/db2>
- Service credentials for the DB2 database. Get the db2 credentials from the IBM cloud by following the steps here: <https://cloud.ibm.com/docs/Db2onCloud?topic=Db2onCloud-getting-started>


Note: Testers in the Build Lab team can use the [DB Warehouse-SuperKnowa](https://cloud.ibm.com/services/dashdb/crn%3Av1%3Abluemix%3Apublic%3Adashdb%3Aus-south%3Aa%2Fe65910fa61ce9072d64902d03f3d4774%3A9a36e55c-43f7-4867-8cba-b09ef55c44f9%3A%3A) instance to load their data and run the pipeline. 

The db2_Ingestion module offers a streamlined method for inserting data from CSV or any delimiter file into db2 to fine-tune text to SQL pipelines. 

1. First, set the following credentials in the expertConfig.ini file under the [**DB2_Credentials**] section:

- **dsn_database**: Name of the database.
- **dsn_uid**: User ID for the database.
- **dsn_pwd**: Password for the database.
- **dsn_hostname**: Hostname or IP address of the database server.
- **dsn_port**: Port number of the database server.
- **dsn_protocol**: Protocol used for communication.
- **dsn_driver**: Driver used for database connection.
**


1. If you don’t have delimited files for your database which also contains golden query dataset, you can use a file from the `/input/dataset` folder from the test env.

    ![Sample dataset](../image/013.png)

1. Now specify the file path, including the file name, in the simpleConfig.ini file under the `DataIngestion` section. Additionally, indicate the table name that needs to be created in the db2 database. If you are using the salary.csv, TheHistoryofBaseball is the right schema. Ensuring the right schema is important as the Golden query dataset contains this information in the column db_id. This is required to run the context retriever and the execution evaluation service.

Note: The table may already exist. please use a different table_name.

    
    #Relative path (from home_dir) of csv file to be ingested in db2 table

    #CSV file for Loading
    #filename = ../input/datasets/people.csv
    filename = input/datasets/salary.csv

    #Schema name - Database
    schema_name = TheHistoryofBaseball

    # Table name for CSV data
    table_name= querycraft_db2_test
    

If the user needs to import a file specifying the delimiter for files other than CSV, the user can adjust the delimiter from the expertConfig.ini file:

`delimiter = ,`

**Usage:**

Run the Data Ingestion module of the QueryCraft pipeline using the runQueryCraft.sh, file with the dataIngestion option after setting the simpleConfig.ini file to insert salary.csv into the querycraft_db2_test table in db2.  

`sh runQueryCraft.sh`

Enter the name of the component you want to run:

`dataIngestion`

You can validate the successful execution of the dataIngestion module from the DB2 UI as well.

![DB2 UI](../image/014.png)


## <a name="_toc1138956454"></a>Option 1.2. SQLite Ingestion
If you using a SQLite database, you can upload the folder containing database dump in .sqlite format to the `/input/`.