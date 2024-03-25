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
from sqlalchemy import create_engine, text

openai.api_key = "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"

df = pd.read_csv("expanded_df_10_chunks.csv")
full_df = pd.read_csv('embedded_resume_data.csv')

model = "text-embedding-3-small"
def generate_embedding(text: str) -> list[float]:
    return openai.embeddings.create(input = [text], model=model).data[0].embedding


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_resumes(search_temp_df, query_key_words, n, pprint=True):
   embedding = generate_embedding(query_key_words)
   search_temp_df.drop('similarities', axis=1, inplace=True, errors='ignore')
   # df['similarities'] = df.resume_embedding.apply(lambda x: cosine_similarity(x, embedding))
   search_temp_df['similarities'] = search_temp_df.ChunkEmbedding.apply(
    lambda x: cosine_similarity(x, embedding)
)
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

def extract_keywords(user_query):
    # Initialize the OpenAI client
    client = OpenAI(api_key="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")  # defaults to os.environ.get("OPENAI_API_KEY")

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
        # Create a Thread
    thread = client.beta.threads.create()

    # Add a message to the Thread
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_query
    )

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
    client = OpenAI(api_key="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")  # defaults to os.environ.get("OPENAI_API_KEY")

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
    df_json = merged_df[['ID', 'Resume_str', 'Category']].to_json(orient='split')
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
            database="yt_demo")
        query = "Select * from " + table
        temp_df = pd.read_sql(query, conn)
        conn.close()
        return temp_df

def to_num_list(string):
    num_list = []
    string2 = string[2:-2]
    string_list = string2.strip('][').split(', ')
    for i in string_list:
        num_list.append(np.float64(i))
    return num_list

def user_question(question, num_results, queryNum):
    if queryNum == 0:
        delete_temp_table()
        print('started non-temp block')
        keywords = extract_keywords(question)
        print(keywords)
        print('keywords extracted')
        temp_df = sql_to_pd("embedded")
        temp_df.drop(columns='ChunkEmbedding', inplace=True)
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

        result_df = search_resumes(temp_df, keywords, 10)
        filtered_df = temp_df[temp_df['ID'].isin(result_df['ID'])]
        print(filtered_df.size)
        create_temp_sql_table(filtered_df)
        result_df = result_df.head(num_results)
        print('resumes searched')
        # print(result_df)
        # return result_df
        print('uploading to chat to be summarized')
        res = summarize_answer(question, result_df)
        print('answer has been summarized')
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

def create_sql_table():
    mysql_user = 'admin'
    mysql_password = 'Admin123'
    mysql_host = "ytdemo.c9m0uu86c4k4.us-east-1.rds.amazonaws.com"  # or your database host
    mysql_database = 'yt_demo'

    # Create the SQLAlchemy engine
    engine = create_engine(f'mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_database}')

    if 'ChunkEmbedding' in df.columns:
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