import pandas as pd
import openai
from openai import OpenAI
import numpy as np
import time
import json
import asyncio
import ray
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import ast
import mysql.connector
from mysql.connector import Error
from sqlalchemy import create_engine, text, exc
from extraTesters import *


#must run instantiate_chunk_sql.py if using a new data set than the one already in place

openai.api_key = ""
display_df = pd.DataFrame()
# df = pd.read_csv("chunked_phd_resumes_500.csv")
full_df = pd.read_csv('resumes_phd.csv')
full_df.rename(columns={'Applicant_ID': 'ID'}, inplace=True)

current_username = ""
updated_queries = []

model = "text-embedding-3-small"
def generate_embedding(text: str) -> list[float]:
    return openai.embeddings.create(input = [text], model=model).data[0].embedding


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_resumes(search_temp_df, query_key_words, n, pprint=True):
   embedding = generate_embedding(query_key_words)
   search_temp_df.drop('similarities', axis=1, inplace=True, errors='ignore')
   # df['similarities'] = df.resume_embedding.apply(lambda x: cosine_similarity(x, embedding))
   search_temp_df['similarities'] = search_temp_df.ChunkEmbedding.apply(lambda x: cosine_similarity(x, embedding))
   print("Size of the search_temp_df DataFrame (rows, columns):", search_temp_df.shape)  # Adds description to the size output
   print("Column names in the search_temp_df DataFrame:", search_temp_df.columns)

   res = search_temp_df.sort_values('similarities', ascending=False).drop_duplicates(subset='ID').head(n)

   return res

def  delete_temp_table():
    mysql_user = 'admin'
    mysql_password = 'Admin123'
    mysql_host = "ytdemo.c9m0uu86c4k4.us-east-1.rds.amazonaws.com"  # or your database host
    mysql_database = 'yt_demo'

    # Create the SQLAlchemy engine
    engine = create_engine(f'mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_database}')
    with engine.connect() as connection:
        connection.execute(text("DROP TABLE IF EXISTS temp_embedded"))

def add_user(username, password):
    mysql_user = 'admin'
    mysql_password = 'Admin123'
    mysql_host = "ytdemo.c9m0uu86c4k4.us-east-1.rds.amazonaws.com"  # or your database host
    mysql_database = 'yt_demo'
    print('creating engine')
    # Create the SQLAlchemy engine
    engine = create_engine(f'mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_database}')

    # Assuming mysql_user, mysql_password, mysql_host, mysql_database, username, and password are defined somewhere
    with engine.connect() as connection:
        # Create the table if it doesn't exist
        connection.execute(text("""
        CREATE TABLE IF NOT EXISTS credentials (
            username VARCHAR(255) UNIQUE,
            password VARCHAR(255),
            queries JSON NOT NULL
        )
        """))
        print('Table created.')

        # Check if the username already exists
        result = connection.execute(text("""
        SELECT username FROM credentials WHERE username = :username
        """), {'username': username}).fetchone()
        print('Checked if username existed.')

        if result:
            return False
        else:
            try:
                # Insert the new user with an empty JSON array for 'queries'
                connection.execute(text("""
                INSERT INTO credentials (username, password, queries) VALUES (:username, :password, '[]')
                """), {'username': username, 'password': password})
                print("User added successfully.")
                return True
            except exc.SQLAlchemyError as e:
                print(f"An error occurred: {e}")


def check_credentials(username, password):
    global current_username
    global updated_queries
    mysql_user = 'admin'
    mysql_password = 'Admin123'
    mysql_host = "ytdemo.c9m0uu86c4k4.us-east-1.rds.amazonaws.com"  # or your database host
    mysql_database = 'yt_demo'
    # Create the SQLAlchemy engine
    engine = create_engine(f'mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_database}')

    with engine.connect() as connection:
        # Attempt to find a row matching the username and password
        result = connection.execute(text("""
        SELECT * FROM credentials WHERE username = :username AND password = :password
        """), {'username': username, 'password': password}).fetchone()

        # Return True if a matching row was found, False otherwise
        if(result is not None):
            current_username = username
            print("current username is " + current_username)
            result = connection.execute(text("""
                    SELECT queries FROM credentials WHERE username = :username
                """), {'username': username}).fetchone()


            # Parse the existing queries JSON, append the new query, and update the row
            current_queries_str = result['queries']
            if current_queries_str is None or current_queries_str == '':
                current_queries = []
            else:
                current_queries = json.loads(current_queries_str)
            # print(type(current_queries))
            # print(current_queries)
        
            updated_queries = current_queries  # Append the new query
        return result is not None

def create_temp_sql_table(temp_table):
    mysql_user = 'admin'
    mysql_password = 'Admin123'
    mysql_host = "ytdemo.c9m0uu86c4k4.us-east-1.rds.amazonaws.com"  # or your database host
    mysql_database = 'yt_demo'

    # Create the SQLAlchemy engine
    engine = create_engine(f'mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_database}')

    if 'ChunkEmbedding' in temp_table.columns:
            temp_table['resume_embedding_json'] = temp_table['ChunkEmbedding'].apply(json.dumps)

    print(temp_table.columns)
    total_rows = len(temp_table)
    for start_row in range(0, total_rows, 750):
        end_row = min(start_row + 750, total_rows)
        df_chunk = temp_table.iloc[start_row:end_row]

        # If it's the first chunk, ensure the table is created
        if start_row == 0:
            pd_to_sql(df_chunk, 'temp_embedded', engine, first_call=True)
        else:
            pd_to_sql(df_chunk, 'temp_embedded', engine)
        print('row' + str(start_row) + ' to row' + str(end_row) + ' have been uploaded')
    print('table uploaded')

def agent(user_query):
    openai.api_key = ""
    client = OpenAI(api_key="")  # defaults to os.environ.get("OPENAI_API_KEY")

    # Create an Assistant
    assistant = client.beta.assistants.create(
        name="mySQL Expert",
        instructions=("""You are an AI designed to analyze user queries. 
                 Your task is to break down the query into parts, categorize each part as either 'lexical' or 'semantic',  
                 and identify logical connectors between parts. 
                 Return a nested array with three elements: 
                 subarray 1. An array of the query parts, 
                 subarray 2. An array indicating whether each part is 'lexical' or 'semantic', 
                 subarray3. An array of the logical connectors ('and', 'or') between parts. 
                Try your very best to only split the query at parts where it says 'and' or 'or'. It's ok to not split the query if it isn't needed. In that case, subarray 1 would only have 1 element, subarray 2 would only have 1 element, and subarray 3 would be empty. 
                 
                Lexical parts are those that can be directly answered with an SQL query on a table with with columns: "First Name", "Last Name", "State", "Email", "Undergrad University", "Masters University", "GPA", "Number of Professional_Research Experiences", "Number of Publications", "GRE Quantitative", "GRE Verbal", "GRE Writing". 
                 Semantic parts require analysis beyond direct SQL queries, such as using cosine similarity. 
                 Return only the nested array, with no additional text or explanation.
                  The example query is : 'Show me candidates with Azure DevOps experience OR knowledge of Microsoft Office Suite AND a GPA above 3.8'
                    Your output would be : [['Candidates with Azure DevOps experience', 'knowledge of Microsoft Office Suite', 'a GPA above 3.8'],
                    ['semantic', 'semantic', 'lexical'], ['OR', 'AND']].
                    Simple sentences like 'Show me students with an interest in lexicography' don't need to be broken into segments.
                    This example would just return an output of [['students with an interest in lexicography'], ['semantic'], []].
                    This is because the input was a simple sentence and no logical operators in it."""
                ),
        model="gpt-4-1106-preview"
    )
    print('assistant created')
        # Create a Thread
    thread = client.beta.threads.create()

    # Add a message to the Thread
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_query
    )
    print('message addded to thread')
    # Create a Run
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="""You are an AI designed to analyze user queries. 
                 Your task is to break down the query into parts, categorize each part as either 'lexical' or 'semantic',  
                 and identify logical connectors between parts. 
                 Return a nested array with three elements: 
                 subarray 1. An array of the query parts, 
                 subarray 2. An array indicating whether each part is 'lexical' or 'semantic', 
                 subarray3. An array of the logical connectors ('and', 'or') between parts. 
                Try your very best to only split the query at parts where it says 'and' or 'or'. It's ok to not split the query if it isn't needed. In that case, subarray 1 would only have 1 element, subarray 2 would only have 1 element, and subarray 3 would be empty. 
                 
                Lexical parts are those that can be directly answered with an SQL query on a table with with columns: "First Name", "Last Name", "State", "Email", "Undergrad University", "Masters University", "GPA", "Number of Professional_Research Experiences", "Number of Publications", "GRE Quantitative", "GRE Verbal", "GRE Writing". 
                 Semantic parts require analysis beyond direct SQL queries, such as using cosine similarity. 
                 Return only the nested array, with no additional text or explanation.
                  The example query is : 'Show me candidates with Azure DevOps experience OR knowledge of Microsoft Office Suite AND a GPA above 3.8'
                    Your output would be : [['Candidates with Azure DevOps experience', 'knowledge of Microsoft Office Suite', 'a GPA above 3.8'],
                    ['semantic', 'semantic', 'lexical'], ['OR', 'AND']].
                    Simple sentences like 'Show me students with an interest in lexicography' don't need to be broken into segments.
                    This example would just return an output of [['students with an interest in lexicography'], ['semantic'], []].
                    This is because the input was a simple sentence and no logical operators in it.""")
    print('waiting for response')
    # Wait for the response
    while run.status != "completed":
        time.sleep(1)  # Sleep for 1 second between checks
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
    # If run completed, retrieve and print messages
    if run.status == "completed":
        print('run complete!')
        messages = client.beta.threads.messages.list(
            thread_id=thread.id,
            order="asc"
        )
    # Initialize a string to hold the concatenated message contents
    all_messages_content = ""

    for message in messages:
        if message.role == "assistant":  # Check if the message is from the assistant
            for text_or_image in message.content:
                if text_or_image.type == 'text':
                    # Append the message text to the all_messages_content string instead of the message object
                    all_messages_content += str(text_or_image.text.value)
    print(all_messages_content)
    nested_array = eval(all_messages_content)

    query_segments = nested_array[0]  # The first level array
    type_array = nested_array[1]  # The second level array
    logic_array = nested_array[2]
    return query_segments,type_array, logic_array

def extract_keywords(user_query):
    # Initialize the OpenAI client
    client = OpenAI(api_key="")  # defaults to os.environ.get("OPENAI_API_KEY")

    # Create an Assistant
    assistant = client.beta.assistants.create(
        name="mySQL Expert",
        instructions=("""You are my assistant. I need you to extract keywords from my query that 
                      would be most effective in using cosine similarity to search through vector
                      embedded resumes and
                      I need you to only return the key words. For example, if I asked 'Show me 
                      college graduates who are interested in management consulting and ' I would want your
                      response to only be 'college, graduate, management, consulting, business, finance' and
                      any other words you think may be helpful in the cosine similarity search"""),
        model="gpt-4-1106-preview"
    )
    print('assistant created')
        # Create a Thread
    thread = client.beta.threads.create()

    # Add a message to the Thread
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_query
    )
    print('message addded to thread')
    # Create a Run
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="""You are my assistant. I need you to extract keywords from my query that 
                      would be most effective in using cosine similarity to search through vector
                      embedded resumes and
                      I need you to only return the key words. For example, if I asked 'Show me 
                      college graduates who are interested in management consulting and ' I would want your
                      response to only be 'college, graduate, management, consulting, business, finance' and
                      any other words you think may be helpful in the cosine similarity search'"""
    )
    print('waiting for response')
    # Wait for the response
    while run.status != "completed":
        time.sleep(1)  # Sleep for 1 second between checks
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
    # If run completed, retrieve and print messages
    if run.status == "completed":
        messages = client.beta.threads.messages.list(
          thread_id=thread.id,
          order="asc"
        )
    # Initialize a string to hold the concatenated message contents
    all_messages_content = ""

    for message in messages:
        if message.role == "assistant":  # Check if the message is from the assistant
            for text_or_image in message.content:
                if text_or_image.type == 'text':
                    # Append the message text to the all_messages_content string instead of the message object
                    all_messages_content += str(text_or_image.text.value)

    # Now, all_messages_content contains all the text content concatenated together

    
    return all_messages_content

def summarize_answer(user_query, result_df):
    global display_df
    client = OpenAI(api_key="")  # defaults to os.environ.get("OPENAI_API_KEY")

    # Create an Assistant
    assistant = client.beta.assistants.create(
        name="mySQL Expert",
        instructions=("You are my assistant. I need you to summarize the following information that was a response to this question: " + user_query +
                      "Try to make the answer as insightful as possible while remaining concise and using only given information as an answer to the question" +
                      "Don't include your own advice in terms of choosing between potential candidates. Keep the answer to around than" +
                      "100 words and definitely less than 150 words. Only have a couple of lines about each candidate" + 
                      "This information is a subset of a larger group, so don't refer to candidates as first or second candidate" +
                      "Reference each applicant by their ID."),
        model="gpt-4-1106-preview"
    )
        # Create a Thread
    thread = client.beta.threads.create()
    print('thread created')
    # Assuming df is your DataFrame and 'column_name' is the name of the column you want to convert
    id_matches = result_df['ID'].values
    full_df.set_index('ID', inplace=True)
    merged_df = full_df.loc[id_matches]
    # Reset index to make 'ID' a column again before converting to JSON
    merged_df.reset_index(inplace=True)
    full_df.reset_index(inplace=True)
    print('full resumes merged in')
    display_df = merged_df[['ID', 'Resume']]
    print('display df instantiated')
    print(display_df)
    df_json = merged_df[['ID', 'Resume']].to_json(orient='split')
    print('converted merged_df to json')
    
    # Add a message to the Thread
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=df_json
    )

    # Create a Run
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions=("You are my assistant. I need you to summarize the following information that was a response to this question: " + user_query +
                      "Try to make the answer as insightful as possible while remaining concise and using only given information as an answer to the question" +
                      "Don't include your own advice in terms of choosing between potential candidates. Keep the answer to around than" +
                      "100 words and definitely less than 150 words. Only have a couple of lines about each candidate" + 
                      "This information is a subset of a larger group, so don't refer to candidates as first or second candidate" +
                      "Reference each applicant by their ID."),
    )
    print('uploaded question, waiting on response')
    # Wait for the response
    while run.status != "completed":
        time.sleep(1)  # Sleep for 1 second between checks
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
    # If run completed, retrieve and print messages
    if run.status == "completed":
        messages = client.beta.threads.messages.list(
          thread_id=thread.id,
          order="asc"
        )
    # Initialize a string to hold the concatenated message contents
    all_messages_content = ""

    for message in messages:
        if message.role == "assistant":  # Check if the message is from the assistant
            for text_or_image in message.content:
                if text_or_image.type == 'text':
                    # Append the message text to the all_messages_content string instead of the message object
                    all_messages_content += str(text_or_image.text.value)

    # Now, all_messages_content contains all the text content concatenated together

    
    return all_messages_content

def execute_sql(query):
    try:
        conn = mysql.connector.connect(
            host="ytdemo.c9m0uu86c4k4.us-east-1.rds.amazonaws.com",
            user="admin",
            password="Admin123",
            database="yt_demo"
        )
        if conn.is_connected():
            print("Successfully connected to the database")
    except Error as e:
        print("Error while connecting to MySQL", e)
    cursor = conn.cursor()
    cursor.execute(query)
    response = []
    # Fetch and print the results
    for row in cursor.fetchall():
            response.append(row)
    # Close the cursor and connection
    cursor.close()
    conn.close()
    return response

def pd_to_sql(df, new_name, engine, first_call=False):
    # Convert 'ChunkEmbedding' to a JSON string if it's not already done
    if 'ChunkEmbedding' in df.columns and isinstance(df['ChunkEmbedding'].iloc[0], (list, np.ndarray)):
        df['resume_embedding_json'] = df['ChunkEmbedding'].apply(json.dumps)
        df = df.drop(columns=['ChunkEmbedding'])

    # Keep only columns that do not contain lists or np.ndarrays
    df = df[[column for column in df.columns if not isinstance(df[column].iloc[0], (list, np.ndarray))]]

    # If it's the first call, replace/create the table. Otherwise, append to it.
    if_exists_action = 'replace' if first_call else 'append'
    df.to_sql(new_name, con=engine, if_exists=if_exists_action, index=False)

def sql_to_pd(table):

        conn = mysql.connector.connect(
            host="ytdemo.c9m0uu86c4k4.us-east-1.rds.amazonaws.com",
            user="admin",
            password="Admin123",
            database="yt_demo",
            compress = True)
        query = "Select * from " + table
        dfs = pd.read_sql(query, conn, chunksize=2000)
    
        # Concatenate chunks into a single DataFrame
        temp_df = pd.concat([chunk for chunk in dfs], ignore_index=True)
        print('temp df size')
        print(temp_df.size)
        conn.close()
        return temp_df

def to_num_list(string):
    num_list = []
    string2 = string[2:-2]
    string_list = string2.strip('][').split(', ')
    for i in string_list:
        num_list.append(np.float64(i))
    return num_list

def merge_dfs(dataframes, operations):
    if not dataframes or len(dataframes) - 1 != len(operations):
        print("Invalid input: array2 must have one less element than array1")
        return None
    if(len(dataframes) == 1):
        return dataframes[0]
    result_df = dataframes[0]  # Initialize the result with the first DataFrame
    for i, operation in enumerate(operations):
        # Merge result_df with the next DataFrame in the list using the specified operation
        result_df = pd.merge(result_df, dataframes[i + 1], how=operation, on='ID')
    result_df = result_df[['ID']].drop_duplicates().sort_values('ID').reset_index(drop=True)
    return result_df

def map_logic_to_merge(operators):
    # Mapping of logic operators to merge operations
    operation_map = {
        "AND": "inner",
        "OR": "outer"
    }
    
    # Convert the list of logic operators to their corresponding merge operations
    # Convert each operator to uppercase for case-insensitive comparison
    merge_operations = [operation_map[op.upper()] for op in operators if op.upper() in operation_map]
    
    return merge_operations

def append_query_to_user(username, query):
    global updated_queries
    mysql_user = 'admin'
    mysql_password = 'Admin123'
    mysql_host = "ytdemo.c9m0uu86c4k4.us-east-1.rds.amazonaws.com"  # or your database host
    mysql_database = 'yt_demo'
    print('creating engine')
    # Create the SQLAlchemy engine
    engine = create_engine(f'mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_database}')
    with engine.connect() as connection:
        # Begin a transaction
        with connection.begin():
            try:
                # Retrieve the current queries array for the user
                result = connection.execute(text("""
                    SELECT queries FROM credentials WHERE username = :username
                """), {'username': username}).fetchone()

                if result:
                    # Parse the existing queries JSON, append the new query, and update the row
                    current_queries_str = result['queries']
                    if current_queries_str is None or current_queries_str == '':
                        current_queries = []
                    else:
                        current_queries = json.loads(current_queries_str)
                    # print(type(current_queries))
                    # print(current_queries)
                
                    updated_queries = current_queries + [query]  # Append the new query
                    # print(f"Username: {username}, Updated Queries: {json.dumps(updated_queries)}")
                    connection.execute(text("""
                        UPDATE credentials
                        SET queries = :updated_queries
                        WHERE username = :username
                    """), {'updated_queries': json.dumps(updated_queries), 'username': username})

                    print("Query appended successfully.")
                    return True
                else:
                    print("Username not found.")
                    return False
            except exc.SQLAlchemyError as e:
                print(f"An error occurred: {e}")
                return False

def get_sql_query(user_query, sql_table):
    openai.api_key = ""
    client = OpenAI(api_key="")  # defaults to os.environ.get("OPENAI_API_KEY")

    # Create an Assistant
    assistant = client.beta.assistants.create(
        name="mySQL Expert",
        instructions=(f"""You are a mySQL expert. Generate mySQL queries to answer questions 
                    based on a database table named '{sql_table}'. The table has the following columns:
                    - ID (INTEGER PRIMARY KEY)
                    - First_Name (TEXT)
                    - Last_Name (TEXT)
                    - State (TEXT)
                    - Email (TEXT)
                    - Undergrad_University (TEXT)
                    - Masters_University (TEXT)
                    - GPA (FLOAT)
                    - Number_of_Professional_Research_Experiences (INT)
                    - Number_of_Publications (INT)
                    - GRE_Quantitative (FLOAT)
                    - GRE_Verbal (FLOAT)
                    - GRE_Writing (FLOAT)
                    When asked to provide information about applicants, use the exact column names from this table.
                    Respond with only the mySQL statement that could be copy pasted
    and executed. ALWAYS SELECT AT LEAST THE ID. Only the query and no extra words or explanations. Only use the exact column names from this table.
    Answers should be formatted similarly to: "SELECT * FROM lexical_table ORDER BY GPA DESC LIMIT 5;"
                    """
                ),
        model="gpt-4-1106-preview"
    )
    print('assistant created')
        # Create a Thread
    thread = client.beta.threads.create()

    # Add a message to the Thread
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_query
    )
    print('message addded to thread')
    # Create a Run
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions=f"""You are a mySQL expert. Generate mySQL queries to answer questions 
                    based on a database table named '{sql_table}'. The table has the following columns:
                    - ID (INTEGER PRIMARY KEY)
                    - First_Name (TEXT)
                    - Last_Name (TEXT)
                    - State (TEXT)
                    - Email (TEXT)
                    - Undergrad_University (TEXT)
                    - Masters_University (TEXT)
                    - GPA (FLOAT)
                    - Number_of_Professional_Research_Experiences (INT)
                    - Number_of_Publications (INT)
                    - GRE_Quantitative (FLOAT)
                    - GRE_Verbal (FLOAT)
                    - GRE_Writing (FLOAT)
                    When asked to provide information about applicants, use the exact column names from this table.
                    Respond with only the mySQL statement that could be copy pasted
    and executed. ALWAYS SELECT AT LEAST THE ID. Only the query and no extra words or explanations. Only use the exact column names from this table
    Answers should be formatted similarly to: "SELECT * FROM lexical_table ORDER BY GPA DESC LIMIT 5;"
                    """)
    print('waiting for response')
    # Wait for the response
    while run.status != "completed":
        time.sleep(1)  # Sleep for 1 second between checks
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
    # If run completed, retrieve and print messages
    if run.status == "completed":
        print('run complete!')
        messages = client.beta.threads.messages.list(
            thread_id=thread.id,
            order="asc"
        )
    # Initialize a string to hold the concatenated message contents
    all_messages_content = ""

    for message in messages:
        if message.role == "assistant":  # Check if the message is from the assistant
            for text_or_image in message.content:
                if text_or_image.type == 'text':
                    # Append the message text to the all_messages_content string instead of the message object
                    all_messages_content += str(text_or_image.text.value)
    print(all_messages_content)
    return all_messages_content

def instantiate():
    df = pd.read_csv('resume_columns_phd copy.csv')
    create_sql_table(df=df, name='lexical_table')
    mysql_user = 'admin'
    mysql_password = 'Admin123'
    mysql_host = "ytdemo.c9m0uu86c4k4.us-east-1.rds.amazonaws.com"  # or your database host
    mysql_database = 'yt_demo'
    print('creating engine')
    # Create the SQLAlchemy engine
    engine = create_engine(f'mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_database}')

    # Assuming mysql_user, mysql_password, mysql_host, mysql_database, username, and password are defined somewhere
    with engine.connect() as connection:

        connection.execute('ALTER TABLE lexical_table ADD PRIMARY KEY (`ID`);')

def execute_chat_sql(sql_query):
    mysql_user = 'admin'
    mysql_password = 'Admin123'
    mysql_host = "ytdemo.c9m0uu86c4k4.us-east-1.rds.amazonaws.com"  # or your database host
    mysql_database = 'yt_demo'
    print('creating engine')
    # Create the SQLAlchemy engine
    engine = create_engine(f'mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_database}')

    sql_ans_df = pd.read_sql_query(sql_query, engine)
    return sql_ans_df

import re

def parse_select_statements(input_string):
    # Regular expression to match patterns that start with 'SELECT' and end with ';'
    pattern = r'SELECT.*?;'
    # Find all matches in the input string
    matches = re.findall(pattern, input_string, re.DOTALL)
    print(matches[0])
    return matches[0]

# Example usage:
# input_string = "Here's a query: SELECT * FROM table1; And another: SELECT column1, column2 FROM table2;"
# print(parse_select_statements(input_string))


def lexical_process(user_query, sql_table):

    return execute_chat_sql(parse_select_statements(get_sql_query(user_query, sql_table)))

def user_question(question, num_results, queryNum):
    full_start = time.time()
    append_query_to_user(current_username, query=question)
    df_list = []
    if queryNum == 0:
        print('trying agent')
        query_segments,type_array, logic_array = agent(user_query=question)
        logic_array = map_logic_to_merge(logic_array)
        print(query_segments)
        print(type_array)
        print(logic_array)
        if 'semantic' in type_array:
                start_time = time.time()

                temp_df = sql_to_pd("embedded")
                temp_df.drop(columns='ChunkEmbedding', inplace=True)

                end_time = time.time()

                duration = end_time - start_time
                print(f"Execution time: {duration} seconds")
                print('Downloaded from sql')
                temp_df['ChunkEmbedding'] = temp_df['resume_embedding_json'].apply(lambda x: to_num_list(x))

                # After conversion, checking the type of the first item in the converted column
                # print(temp_df['ChunkEmbedding'].iloc[0])
                # print("After conversion:", type(temp_df['ChunkEmbedding'].iloc[0]))
                print('conversion completed')
        for i in range(len(query_segments)):
            question = query_segments[i]            
            if type_array[i] == 'semantic':
                print('entered semantic block')
                delete_temp_table()
                print('started non-temp block')
                keywords = extract_keywords(question)
                print(keywords)
                print('keywords extracted')
                
                # Just before conversion
                # print("Before conversion:", type(temp_df['resume_embedding_json'].iloc[0]))
                # print(temp_df['resume_embedding_json'].iloc[0])

                
                result_df = search_resumes(temp_df, keywords, 10)
                filtered_df = temp_df[temp_df['ID'].isin(result_df['ID'])]
                print(filtered_df.size)
                create_temp_sql_table(filtered_df)
                # result_df = result_df.head(num_results)
                df_list.append(result_df)
                setdf2(result_df)
            if type_array[i] == 'lexical':
                print('entered lexical block')
                result_df = lexical_process(question, 'lexical_table')
                print('lexical process finished')
                # filtered_df = temp_df[temp_df['ID'].isin(result_df['ID'])]
                # print(filtered_df.size)
                # create_temp_sql_table(filtered_df)
                # result_df = result_df.head(num_results)
                df_list.append(result_df)
                setdf1(result_df)
        print('resumes searched')
        # print(result_df)
        # return result_df
        # for df in df_list:
        #     print(df)
        print('using logic to merge tables')
        final_result_df = merge_dfs(df_list, logic_array)
        final_result_df = final_result_df.head(num_results)
        print('tables merged - uploading to chat to be summarized')
        res = summarize_answer(question, final_result_df)
        print('answer has been summarized')
        full_end_time = time.time()

        full_duration = full_end_time - full_start
        print(f"Total execution time: {full_duration} seconds")
    else:
        print('started temp block')
        keywords = extract_keywords(question)
        print(keywords)
        print('keywords extracted')
        temp_df = sql_to_pd("temp_embedded")
        # temp_df.drop(columns='ChunkEmbedding', inplace=True)
        print('Downloaded from sql')
        # Just before conversion
        # print("Before conversion:", type(temp_df['resume_embedding_json'].iloc[0]))
        # print(temp_df['resume_embedding_json'].iloc[0])

        # Applying the conversion
        temp_df['ChunkEmbedding'] = temp_df['resume_embedding_json'].apply(lambda x: to_num_list(x))

        # After conversion, checking the type of the first item in the converted column
        # print(temp_df['ChunkEmbedding'].iloc[0])
        # print("After conversion:", type(temp_df['ChunkEmbedding'].iloc[0]))
        print('conversion completed')

        result_df = search_resumes(temp_df, keywords, num_results)
        print('resumes searched')
        print('temp res df')
        # return result_df
        print(result_df)
        print('uploading to chat to be summarized')

        res = summarize_answer(question, result_df)
        print('answer has been summarized')

    return res

def create_sql_table(df):
    print('hi?')
    mysql_user = 'admin'
    mysql_password = 'Admin123'
    mysql_host = "ytdemo.c9m0uu86c4k4.us-east-1.rds.amazonaws.com"  # or your database host
    mysql_database = 'yt_demo'

    # Create the SQLAlchemy engine
    engine = create_engine(f'mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_database}')

    if 'ChunkEmbedding' in df.columns:
            print('good')
            df['resume_embedding_json'] = df['ChunkEmbedding'].apply(json.dumps)

    print(df.columns)
    total_rows = len(df)
    for start_row in range(0, total_rows, 750):
        end_row = min(start_row + 750, total_rows)
        df_chunk = df.iloc[start_row:end_row]

        # If it's the first chunk, ensure the table is created
        if start_row == 0:
            pd_to_sql(df_chunk, 'embedded', engine, first_call=True)
        else:
            pd_to_sql(df_chunk, 'embedded', engine)
        print('row' + str(start_row) + ' to row' + str(end_row) + ' have been uploaded')
    print('table uploaded')

def get_display_df():
    global display_df
    return display_df

def get_prev_queries():
    global updated_queries
    # print('queries are below')
    # print(updated_queries)
    return updated_queries

