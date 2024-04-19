
# <a name="_toc890332375"></a>Step 5. Query Correction
The Query correction service is used to correct the generated queries before you run it against your database. 

The output of the inference service is dynamically read as the input of the Query correction service and the same CSV is updated.

The input for the query correction can be configured (if required) in the QueryCorrection section of the simpleConfig.ini file:

`input_dataset = ${Default:home_dir}output/inference/${Default:exp_name}_inference.csv`

To start the query correction service, run the command below.

`sh runQueryCraft.sh`

Enter the name of the component you want to run:

`querycorrection`