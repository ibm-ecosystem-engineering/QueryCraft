This repository contains the following scripts and codes

- **superConfig.ini**: A super configuration file to manage basic level settings for data ingestion, context retriever, fine-tuning, inference, query-correction, evaluation, and the query analysis dashboard among other services.
- **config.ini**: A configuration file to manage all settings for data ingestion, context retriever, fine-tuning, inference, query-correction, evaluation, and the query analysis dashboard among other services.
- **db2_connector.py**: This script provides global IBM DB2 connection for all other services.
- **db2_ingestion.py**: This script is used to insert the data into DB2 from a CSV file or any other delimiter file.
- **context_retriever.py**: This script extracts context directly from the SQLite database to inform and improve the accuracy of generated queries.
- **finetune.py**: Fine-tune your LLM specifically for the text-to-SQL task, optimizing its performance for your unique requirements.
- **inference.py**: Run the inference pipeline using either a pre-trained or a fine-tuned model to generate SQL queries from natural language inputs.
- **query_correction.py**: A script dedicated to correcting the syntax of generated SQL queries.
- **ex_evaluator.py**: A script dedicated to calculating the execution accuracy of generated SQL queries against an SQLite database, ensuring that your fine-tuned model performs optimally.
- **pipeline_result_csv_gen.py**: Extracts details of fine-tuning experiments and saves them to a CSV file, facilitating in-depth analysis through the query analysis dashboard.
- **streamlit_query_analysis_dashboard.py**: A Streamlit application that provides a comprehensive dashboard for analyzing the results of your fine-tuned model and conducting comparative analyses.