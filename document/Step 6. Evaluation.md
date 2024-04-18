
# <a name="_toc577../image/012548"></a>Step 6. Evaluation

The evaluation service will run the corrected queries against your database and compare the results of the generated query with the golden query to calculate execution accuracy. You can configure the evaluation service using the superConfig.ini file.



The output of the QueryCorrection service is dynamically read as the input of the Query correction service and the same CSV is updated.

## <a name="_toc1369103650"></a>Option 6.1 Evaluation on the db2 database:

Configure the Evaluation service by updating the EXEvaluator section of the superConfig file for the following parameters:

1. Set the database type for db2.

   `db_type = ‘db2’`

1. Validate the absolute path of the file on which you want to get an execution accuracy score.

    `input_dataset = ${Default:home_dir}output/inference/${Default:exp_name}_inference.csv`

Note: You must have completed the Step 1. Data Ingestion section for DB2.

## <a name="_toc858104986"></a>Option 6.2 Evaluation on sqlite:

Configure the Evaluation service by updating the EXEvaluator section of the superConfig file for the following parameters:

1. Set the database type for db2.

    `db_type = 'sqlite'`

1. Database path: Provide the relative path (from home_dir) to the folder containing the database dump in .sqlite format. Example for Spider SQL database:

    `input_database_folder=${Default:home_dir}input/spider/database/`

1. Inference dataset path: Validate the absolute path of the file on which you want to get an execution accuracy score.

    `input_dataset = ${Default:home_dir}output/inference/${Default:exp_name}_inference.csv`



Evaluate the performance of your model against the SQLite database or DB2 by running the below command:

`sh runQueryCraft.sh`

Enter the name of the component you want to run:

`evaluation`