
# <a name="_toc1725873645"></a>Step 2. Context Retriever

The Context Retriever module offers a convenient solution for accessing context information like DDL schema from both SQLite and db2 databases. 

Configure the ContextRetriever section of superConfig.ini. 

1. input_database_folder: Relative path to the folder containing database dump in .sqlite format. This is required only when using SQLite database
1. input_data_file: Relative path to the golden dataset (CSV file) with columns: question, query, and db_id
1. db_type:  Determines the data source SQLite or db2.


## <a name="_toc2../image/010877930"></a>Option 2.1 Context Retrieval from db2:

For the db2 context retriever, there's no requirement for an input database file like SQLite, as it directly extracts the DDL schema from Db2 tables. Instead, you need to upload the Golden query dataset (input_data_file) for Db2. This Golden query dataset should be uploaded to the input/datasets/ folder. Here is a sample input_data_file named  kaggleDBQASample.csv, and ensure that db_type is mentioned as **db2**.

```
input_database_folder = 
input_data_file = input/datasets/ kaggleDBQASample.csv
db_type = db2
```

## <a name="_toc481704433"></a>Option 2.2 Context Retrieval from SQLite:

For SQLite, the context retriever service expects two inputs:

1. Input Data with golden queries in the form of a CSV file with the following columns: question, query, and db_id
1. Database folder containing .sqlite files.

The output of the context retriever service is a CSV file with the following columns: question, query, db_id, and context. Here, the context includes the DDL schema for the tables in db_id. 

Configure the input data path, database input file, and database type (referred to as db_type) from the superConfig.ini file. The db_type parameter determines the data source for context retrieval, whether it's SQLite or db2.

If you do not have a dataset of your own, we provide either of the two datasets provided as part of this repository—the Spider dataset or the KaggleDBQA dataset.

To use Spider dataset, configure the ContextRetriever section of SuperConfig.ini as:

```
input_database_folder =input/spider/database/
input_data_file = input/datasets/spider.csv
db_type = sqlite
```

To use the KaggleDBQA dataset configure it into ContextRetriever section of SuperConfig.ini as:

```
input_database_folder =input/kaggleDBQA/database/
input_data_file = input/datasets/kaggleDBQA.csv
db_type = sqlite
```

From the example above, you can specify either Spider or KaggleDBQA in the superConfig.ini file.

After updating the **superConfig.ini** as mentioned above, execute the context retriever using the following command.

`sh runQueryCraft.sh`

Enter the name of the component you want to run:

`contextRetriever`

The retrieved context file will be generated in the directory input/datasets/, with the filename exp_name_contextRetriever.csv.

![Instruct dataset with golden queries](../image/015.png)