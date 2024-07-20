import streamlit as st
from pydantic import BaseModel
import requests
import json
import logging
import pyodbc
import os
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sqlparse
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from vector import retrieve_data_by_ids, query_schema_collection, query_question_collection, question_collection, schema_collection
import re
import sqlite3
import subprocess
import os
# Load environment variables from .env file
load_dotenv()
import gdown
import os

def download_database():
    url = 'https://drive.google.com/uc?id=1C-SIp6ifyZiylOK5nSTq-zPpgHw4SzH_'
    output = 'AdventureWorksqlite_db.db'
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)


def create_prompt(question):
    queried_question_ids = query_question_collection(question_collection, question)
    queried_schema_ids = query_schema_collection(schema_collection, question)
    sql_queries = retrieve_data_by_ids('question.csv', queried_question_ids, 2)
    schemas = retrieve_data_by_ids('schema.csv', queried_schema_ids, 2)
    sql_queries_f = "\n\n".join(sql_queries)
    schema_queries_f = "\n\n".join(schemas)
    
    prompt = f"""
    --
    You are an expert in writing SQLite Queries, tasked with generating SQLite queries based on user questions about data stored in various tables in an SQLite database.
    Schema Information which may be relevant:
    {schema_queries_f}
    Example SQLite Query which is more relevant:
    {sql_queries_f}
    Given a user's question about this data, write a valid SQLite query that accurately extracts or calculates the requested information from these tables and adheres to SQL best practices, optimizing for readability and performance where applicable.
    Here are some tips for writing SQLite queries:
    - Ensure all tables referenced are from your SQLite database.
    - Always use aliases for tables.
    - SQLite functions like `date()` or `strftime()` can be used for date calculations.
    - For constructing SQLite queries related to time intervals like "this year" or "this month," use the `strftime()` function to dynamically calculate the appropriate date ranges.
    - Aggregated fields like `COUNT(*)` must be appropriately named.
    Question:
    {question}
    Reminder: Generate an SQLite query to answer the question:
    - Use only those table names which are provided above in Example SQLite Query and Schema Information, do not take irrelevant table names.
    - Do not apply Join Condition unnecessarily between the tables.
    - Respond as a valid JSON Document.
    - [Best] If the question can be answered with the available tables: {{"sql": "<sql here>"}}
    - If the question cannot be answered with the available tables: {{"error": "<explanation here>"}}
    - Ensure that the entire output is returned on only one single line.
    - Keep your query as simple and straightforward as possible; do not use subqueries.
    """
    return prompt


def create_plotly_chart_prompt(sql_query, question, schema_info):
    prompt = f"""
    --
    Schema Information which may be relevant:
    {schema_info}

    Given the following SQL query:
    {sql_query}

    Generate Python code to create a Plotly chart that visualizes the data. The chart should be relevant to the user's question:
    Question: {question}
    The code should:
    - Assume the data is already loaded into a DataFrame named 'results_df'
    - Create a Plotly chart that is relevant to the data and the question
    - Display the chart using Streamlit's st.plotly_chart
    - Respond with only the Python code, and no additional text or comments
    """
    return prompt

def create_reflection_prompt_for_plotly(question, error_message):
    return f"""
    --
    There was an error while generating the Plotly chart code. The following error was encountered:
    {error_message}
    Please regenerate the Python code to create a Plotly chart that visualizes the data related to the question:
    Question: {question}
    The code should:
    - Assume the data is already loaded into a DataFrame named 'results_df'
    - Create a Plotly chart that is relevant to the data and the question
    - Display the chart using Streamlit's st.plotly_chart
    - Respond with only the Python code, and no additional text or comments
    - Ensure no syntax errors like the one previously encountered occur.
    """

def chat_with_groq(client, prompt, model):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )
    return completion.choices[0].message.content

def get_json_output(llm_response):
    llm_response_no_escape = llm_response.replace('\\n', ' ').replace('\n', ' ').replace('\\', '').strip()
    try:
        open_idx = llm_response_no_escape.index('{')
        close_idx = llm_response_no_escape.rindex('}') + 1
        cleaned_result = llm_response_no_escape[open_idx:close_idx]
        json_result = json.loads(cleaned_result)
        if 'sql' in json_result:
            query = json_result['sql']
            return True, sqlparse.format(query, reindent=True, keyword_case='upper')
        elif 'error' in json_result:
            return False, json_result['error']
    except (ValueError, json.JSONDecodeError):
        return False, "No valid JSON content found in response."
    return False, "Response format is incorrect or unexpected."

# def execute_sql_query(query):
#     db_path = "AdventureWorksqlite_db.db"  # Path to the SQLite database file
#     conn = None
#     try:
#         conn = sqlite3.connect(db_path)  # Connect to the SQLite database
#         df = pd.read_sql_query(query, conn)  # Execute the query using pandas
#         return df, None
#     except Exception as e:
#         error_message = str(e)
#         print(f"Failed to execute query: {error_message}")
#         return pd.DataFrame(), error_message
#     finally:
#         if conn:
#             conn.close() 

def execute_sql_query(query):
    db_path = "AdventureWorksqlite_db.db"  # Path to the SQLite database file
    download_database()  # Ensure the latest database file is downloaded
    conn = None
    try:
        conn = sqlite3.connect(db_path)  # Connect to the SQLite database
        df = pd.read_sql_query(query, conn)  # Execute the query using pandas
        return df, None
    except Exception as e:
        error_message = str(e)
        print(f"Failed to execute query: {error_message}")
        return pd.DataFrame(), error_message
    finally:
        if conn:
            conn.close()

def get_reflection(client, full_prompt, error_message, model):
    reflection_prompt = f"""
    You were given the following prompt:
    {full_prompt}
    This was the error encountered:
    {error_message}
    There was an error with the response, either in the query itself.
    Ensure that the following rules are satisfied when correcting your response:
    1. SQL is valid SQLite, given the provided metadata and SQLite querying rules.
    2. The query SPECIFICALLY references the correct tables: 'EmployeePayHistory', 'SalesOrderHeaderSalesReason', 'SalesPerson', 'Illustration', 'JobCandidate', 'Location', 'Password', 'SalesPersonQuotaHistory', 'Person', 'SalesReason', 'SalesTaxRate', 'PersonCreditCard', 'PersonPhone', 'SalesTerritory', 'PhoneNumberType', 'Product', 'SalesTerritoryHistory', 'ScrapReason', 'Shift', 'ProductCategory', 'ShipMethod', 'ProductCostHistory', 'ProductDescription', 'ShoppingCartItem', 'ProductDocument', 'DatabaseLog', 'ProductInventory', 'SpecialOffer', 'ErrorLog', 'ProductListPriceHistory', 'Address', 'SpecialOfferProduct', 'ProductModel', 'AddressType', 'StateProvince', 'ProductModelIllustration', 'AWBuildVersion', 'ProductModelProductDescriptionCulture', 'BillOfMaterials', 'Store', 'ProductPhoto', 'ProductProductPhoto', 'TransactionHistory', 'ProductReview', 'BusinessEntity', 'TransactionHistoryArchive', 'ProductSubcategory', 'BusinessEntityAddress', 'ProductVendor', 'BusinessEntityContact', 'UnitMeasure', 'Vendor', 'ContactType', 'CountryRegionCurrency', 'CountryRegion', 'WorkOrder', 'PurchaseOrderDetail', 'CreditCard', 'Culture', 'WorkOrderRouting', 'Currency', 'PurchaseOrderHeader', 'CurrencyRate', 'Customer', 'Department', 'Document', 'SalesOrderDetail', 'EmailAddress', 'Employee', 'SalesOrderHeader' and those tables are properly aliased? (this is the most likely cause of failure)
    3. Response is in the correct format ({{"sql": "<sql_here>"}} or {{"error": "<explanation here>"}}) with no additional text?
    4. All fields are appropriately named.
    5. There are no unnecessary sub-queries.
    6. ALL TABLES are aliased (extremely important). 
    Rewrite the response and respond ONLY with the valid output format with no additional commentary.
    """
    return chat_with_groq(client, reflection_prompt, model)


def get_summarization(client, user_question, df, model, additional_context):
    prompt = f'''
    A user asked the following question pertaining to local database tables:
    {user_question}
    To answer the question, a dataframe was returned:
    Dataframe:
    {df}
    In a few sentences, summarize the data in the table as it pertains to the original user question. Avoid qualifiers like "based on the data" and do not comment on the structure or metadata of the table itself.
    '''
    if additional_context != '':
        prompt += f'''
        The user has provided this additional context:
        {additional_context}
        '''
    return chat_with_groq(client, prompt, model)

def extract_python_code(response_text):
    match = re.search(r"```python(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return response_text.strip()

def execute_plotly_code(plotly_code, df):
    # Remove Markdown backticks if present
    cleaned_code = plotly_code.replace('```', '').strip()

    # Full Python code including necessary imports and DataFrame assignment
    full_code = f"""
import pandas as pd
import plotly.express as px
import streamlit as st

# Assigning the DataFrame from the function argument
results_df = df

# Plotly code to be executed
{cleaned_code}
"""

    # Define the execution environment
    exec_env = {
        'px': px,        # Ensure Plotly express is available
        'st': st,        # Ensure Streamlit is available
        'pd': pd,        # Ensure Pandas is available
        'df': df,        # Passing the DataFrame directly
        'results_df': df # Making the DataFrame available as 'results_df'
    }

    try:
        # Execute the Python code within the defined environment
        exec(full_code, exec_env)
    except SyntaxError as e:
        st.error(f"SyntaxError in generated code: {e}")
        st.code(full_code, language='python')  # Display the problematic code for better debugging
    except Exception as e:
        st.error(f"Error in executing Plotly code: {e}")
        st.code(full_code, language='python')  # Display the problematic code for better debugging


def main():
    # Configure page to use the full screen width
    # Check if the 'vectordb' folder does not exist
    st.set_page_config(layout="wide")

    # Check if the 'vectordb' folder does not exist
    if not os.path.exists('vectordb'):
        try:
            # If 'vectordb' does not exist, run vector.py
            subprocess.run(["python", "vector.py"], check=True)
            st.success("vector.py executed successfully!")
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to execute vector.py: {e}")
            return  # Optionally return early if the script is critical
    else:
        ""

   
    st.title("Ask Your Database")
    additional_context = "summarize the above info"
    # HTML and CSS for the icon
    icon_html = """
    <style>
    .footer {
        position: fixed;
        right: 10px;
        bottom: 10px;
        z-index: 1000;
    }
    </style>
    <div class="footer">
        <a href="https://drive.google.com/file/d/1__59bj9k3un-N_mzOF3R1lp23MHpe7KS/view?usp=sharing" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/8/87/PDF_file_icon.svg" alt="ERD PDF" width="50" height="50">
        </a>
    </div>
    """
    st.markdown(icon_html, unsafe_allow_html=True)
     # Predefined questions for the dropdown, with an additional option to enter a custom question
    questions = [
        "What are the top 10 products by sales amount?",
        "What is the total number of sales orders grouped by month and year?",
        "What is the total freight cost for each shipping method shown for EACH month and year, x axis should have year and month format",
        "Who are the top 5 salespeople by total sales amount?, grouped by product type",
        "How is the customer base distributed across different sales territories",
        "What is the total revenue generated from online orders and order count",
        "What are the top 10 total sales and total quantity sold for each product?",
        "What is the total revenue generated from each product category?",
        "What is the total tax amount collected for each sales territory, give territory name?"
    ]

    question_selected = st.selectbox("Choose a question or enter your own below:", ["-- Enter your own question --"] + questions)
    
    if question_selected == "-- Enter your own question --":
        user_question = st.text_input("Enter your question here:", value="", key="custom_question")
    else:
        user_question = question_selected

    if st.button("Analyze"):
        if user_question:
            # Initialize the progress bar
            progress_bar = st.progress(0)
            # Setup and authenticate with Groq API
            groq_api_key = os.getenv("GROQ_API_KEY")
            client = Groq(api_key=groq_api_key, base_url=os.getenv("GROQ_BASE_URL"))
            model = "llama3-70b-8192"
            progress_bar.progress(10)  # Update progress after setup

            # Generate the SQL query prompt and process the response
            full_prompt = create_prompt(user_question)
            print(f"This is the full_prompt {full_prompt}")
            progress_bar.progress(20)  # Update progress after generating prompt
            llm_response = chat_with_groq(client, full_prompt, model)
            progress_bar.progress(30)  # Update progress after getting initial response

            # Initialize columns for layout, evenly split
            col1, col2 = st.columns(2)  # Two equal columns
            valid_sql_response = False
            max_retries = 3
            attempts = 0

            while not valid_sql_response and attempts < max_retries:
                is_sql, result = get_json_output(llm_response)
                print(f"This is the iteration number {attempts} and this is the sql query {result}")
                progress_bar.progress(40 + 10 * attempts)

                if is_sql:
                    results_df, error = execute_sql_query(result)
                    progress_bar.progress(60 + 10 * attempts)

                    if not results_df.empty:
                        valid_sql_response = True
                        schema_info = '\n\n'.join(retrieve_data_by_ids('schema.csv', query_schema_collection(schema_collection, user_question), 2))

                        with col1:
                            st.markdown("```sql\n" + result + "\n```")
                            summarization = get_summarization(client, user_question, results_df, model, additional_context)
                            st.write(summarization.replace('$','\\$'))
                            st.dataframe(results_df)

                        plotly_attempts = 0
                        while plotly_attempts < max_retries:
                            plotly_prompt = create_plotly_chart_prompt(result, user_question, schema_info)
                            plotly_response = chat_with_groq(client, plotly_prompt, model)
                            plotly_code = extract_python_code(plotly_response)

                            try:
                                with col2:
                                    execute_plotly_code(plotly_code, results_df)
                                break  # Successful Plotly execution
                            except Exception as e:
                                plotly_attempts += 1
                                print(f"Plotly chart attempt {plotly_attempts} of {max_retries}. Error: {str(e)}")
                                if plotly_attempts < max_retries:
                                    plotly_response = create_reflection_prompt_for_plotly(user_question, str(e))
                                    plotly_response = chat_with_groq(client, plotly_response, model)
                                else:
                                    st.error("Failed to generate valid Plotly code after multiple attempts.")

                        progress_bar.progress(100)  # Complete the progress bar
                        break  # Exit SQL loop if valid response processed
                    else:
                        error_message = f"SQL Execution Error: {error}"
                        print(error_message)
                        if attempts < max_retries - 1:
                            llm_response = get_reflection(client, full_prompt, error, model)

                else:
                    print("Invalid SQL response. Attempting reflection.")
                    if attempts < max_retries - 1:
                        llm_response = get_reflection(client, full_prompt, result, model)
                attempts += 1

            if not valid_sql_response:
                st.error("Failed to generate a valid SQL response after multiple attempts.")
                progress_bar.empty()

        else:
            st.error("Please enter your question.")
            progress_bar.empty()

if __name__ == "__main__":
    main()
