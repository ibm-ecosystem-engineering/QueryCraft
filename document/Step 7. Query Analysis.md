
# <a name="_toc766523960"></a>Step 7. Query Analysis Dashboard

Query analysis dashboard service can be used to analyze the generated queries, categories of the incorrect queries, evaluation results and comparison between the results of multiple LLMs. 

The output of the Evaluation service is dynamically read as the input of the QueryAnalysisDashboard.

The input for the query correction can be configured (if required) in the QueryAnalysisDashboard section of the superConfig.ini file:

`folder_name =output/evalResults/`

Launch the dashboard for visual analysis of your fine-tuning experiments and generated SQL queries:

`sh runQueryCraft.sh`

Enter the name of the component you want to run:

`queryanalysisDashboard`

Use one of the following ports for exposing your dashboard: 8506-8510.

Please ensure that the URL is not already under use for the port you are selecting: [http://169.46.68.130:port_number]()

Ex: <http://169.46.68.130:85../image/01>

Then you can run the following command with the **right port number** to run the dashboard and access it in your browser.

```
cd code
streamlit run streamlit_query_analysis_dashboard.py --server.port 8502 --server.fileWatcherType none
```