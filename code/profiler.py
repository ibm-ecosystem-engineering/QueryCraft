import pandas as pd
import re
import numpy as np
class ProfileAnalyzer:
    def __init__(self, schema):
        self.schema = schema

    def analyze_query(self, query):
        analysis = {
            "has_join": bool(re.search(r"\bJOIN\b", query, flags=re.IGNORECASE)),
            "has_where": bool(re.search(r"\bWHERE\b", query, flags=re.IGNORECASE)),
            "has_groupby": bool(re.search(r"\bGROUP BY\b", query, flags=re.IGNORECASE)),
            "has_aggregate": any(word in query for word in ["COUNT", "SUM", "AVG", "MAX", "MIN"]),
            "nesting_level": 0,
            "schema_aware": False
        }

        subquery_keywords = ["SELECT", "WHERE"]
        for keyword in subquery_keywords:
            open_subqueries = query.count(f"{keyword}")
            analysis["nesting_level"] += max(int(open_subqueries / 2), 0)

        return analysis

    def extract_table_names(self, query):
        tables = re.findall(r"\bFROM\s+([^\s,;]+)", query, flags=re.IGNORECASE)
        return list(set(tables))

    def extract_columns(self, query):
        columns = re.findall(r"\bSELECT\s+(.+?)\bFROM", query, flags=re.DOTALL | re.IGNORECASE)
        if columns:
            return [col.strip() for col in columns[0].split(",")]
        else:
            return []

    def check_schema_compliance(self, query):
        tables = self.extract_table_names(query)
        columns = self.extract_columns(query)

        for table in tables:
            if table not in self.schema:
                return False

        for column in columns:
            if "." in column:
                table, col = column.split(".")
                if table not in self.schema or col not in self.schema[table]:
                    return False
            else:
                if not any(column in cols for cols in self.schema.values()):
                    return False

        return True

    def analyze_dataframe(self, df):
        df["query_analysis"] = df["sql_query"].apply(self.analyze_query)
        df["has_join"] = df["query_analysis"].apply(lambda x: x["has_join"])
        df["has_where"] = df["query_analysis"].apply(lambda x: x["has_where"])
        df["has_groupby"] = df["query_analysis"].apply(lambda x: x["has_groupby"])
        df["has_aggregate"] = df["query_analysis"].apply(lambda x: x["has_aggregate"])

        df["table_names"] = df["sql_query"].apply(self.extract_table_names)
        all_tables = df["table_names"].explode().unique()
        all_tables = [table for table in all_tables if table]

        df["columns"] = df["sql_query"].apply(self.extract_columns)
        all_columns = df["columns"].explode().unique()
        all_columns = [col for col in all_columns if col]

        df["num_columns"] = df["columns"].apply(len)
        num_columns_distribution = df["num_columns"].value_counts().sort_index()

        df["nesting_level"] = df["query_analysis"].apply(lambda x: x["nesting_level"])
        nesting_level_distribution = df["nesting_level"].value_counts().sort_index()

        df["schema_aware"] = df["sql_query"].apply(lambda query: self.check_schema_compliance(query))

        return df, all_tables, all_columns, num_columns_distribution, nesting_level_distribution



if __name__ == "__main__":
    schema = {
        "Contract": [
            "Expiration Date", "Supplier", "ID", "TCV", "Term Type",
            "Reporting Currency", "Status", "Title", "Document Type",
            "Effective Date", "Functions", "Services", "Regions", "Countries",
            "Time Zone", "Currencies", "Agreement Type", "Name", "Source Name/Title"
        ],
        "Contract Draft Request": [
            "ID", "Title", "Suppliers", "ESignature Status", "Source Name/Title",
            "Total Deviations", "Effective Date", "TCV", "Paper Type", "Status",
            "Regions", "Countries", "Functions", "Services", "Templates",
            "Counterparty Type", "Agreement Type", "Expiration Date", "Multilingual",
            "No Touch Contract"
        ],
        "CO/CDR": [
            "Created On", "Created By", "Counterparty", "Reporting Date"
        ]
    }

    analyzer = ProfileAnalyzer(schema)

    df = pd.read_csv(r"./eval_query.csv")

    df, all_tables, all_columns, num_columns_distribution, nesting_level_distribution = analyzer.analyze_dataframe(df)

    print("Query Type Counts:")
    print(df[["has_join", "has_where", "has_groupby", "has_aggregate"]].mean())  # Average presence of each query type

    print("\nUnique Table Names:")
    print(all_tables)

    print("\nUnique Columns Across All Queries:")
    print(all_columns)

    print("\nDistribution of Number of Columns Referenced:")
    print(num_columns_distribution)

    print("\nDistribution of Nesting Levels:")
    print(nesting_level_distribution)

    print("\nSchema Awareness Check:")
    print(df["schema_aware"].value_counts())
