

# <a name="_toc523711792"></a>Step 8.  Run pipeline (All)

To run all components together, you can change the required parameters in `simpleConfig.ini`. You must set the default path as shown in the designated section below. 

[Finetune] 

`train_dataset=${Default:home_dir}input/datasets/${Default:exp_name}_contextRetriever.csv` 



[Inference] 

`input_dataset = input/datasets/${Default:exp_name}_validSet.csv`



[QueryCorrection] 

`input_dataset = ${Default:home_dir}output/inference/${Default:exp_name}_inference.csv` 



[EXEvaluator] 

`input_dataset = ${Default:home_dir}output/inference/${Default:exp_name}_inference.csv`


Run the below command:

`sh runQueryCraft.sh`

Enter the command 

`all`
