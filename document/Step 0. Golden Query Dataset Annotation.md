# <a name="_toc1843293825"></a>Step 0. Golden Query Dataset (Instruct dataset) Annotation

There are three options for using your dataset to finetune/evaluate the Text to SQL (QueryCraft) pipeline:

1. Bring your dataset with golden queries in the following format: question, query, and db_id. Instruction for ingesting the dataset is provided in the next Step 1.
1. Curate the golden query dataset using our annotation tool: <https://annotator.superknowa.tsglwatson.buildlab.cloud/>
1. Use the example datasets provided below for testing: Spider and KaggleDBQA

   Unzip the example datasets using the command:

   ```cd ./input

   unzip spider.zip

   unzip KaggleDBQA.zip

   cd ..
   ```

## <a name="_toc548496318"></a>Golden Query Annotation:
1. Go to our annotation tool. <https://annotator.superknowa.tsglwatson.buildlab.cloud/>

![Data annotator view](../image/011.png)

2. Click on the Instruction Manual and follow the instructions for curating the golden queries dataset. <https://annotator.superknowa.tsglwatson.buildlab.cloud/documentation>

![Data annotation instruction manual](../image/012.png)