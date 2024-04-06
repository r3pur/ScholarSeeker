from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import os
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

from main import *



app = Flask(__name__)
app.secret_key = 'randomTestplsWork'

# Initialize global variables
chat_history = []
queryNum = 0  
signed_in = False

# # Define the directory where uploaded files will be saved
# UPLOAD_FOLDER = 'uploads'  # Replace this with the actual path on your server
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    global signed_in
    if signed_in:
        global chat_history, queryNum  # Refer to the global variables
        user_query = request.json['message']
        num_input = int(request.form.get('num', 2))  # Default to 1 if not provided
        response = user_question(user_query, num_input, queryNum=queryNum)
        chat_history = chat_history[-5:]  # Keep last 6 messages
        chat_history += [{'user': user_query}, {'scholarseeker': response}]
        queryNum += 1  # Increment queryNum by 1 for each submission
        chat_history = []  
        

        return jsonify({'message': response}), 201
    else:
        print('in right spot')
        return jsonify({'message': 'Sign in before using'}), 400

@app.route('/reset', methods=['POST'])
def reset():
    print('reset button worked')
    global chat_history, queryNum
    chat_history = []  # Clear the chat history
    queryNum = 0  # Reset the query number to 0
    return jsonify({'message': 'Chat history and query number have been reset.'})

    # if request.method == 'POST':
    #     action = request.form.get('action')  # Safely get the action value
        
    #     if action == 'submit':
    #         user_query = request.form.get('query', '')
    #         num_input = int(request.form.get('num', 1))  # Default to 1 if not provided
    #         response = user_question(user_query, num_input, queryNum=queryNum)
    #         output = response
    #         chat_history = chat_history[-5:]  # Keep last 6 messages
    #         chat_history += [{'user': user_query}, {'scholarseeker': response}]
    #         queryNum += 1  # Increment queryNum by 1 for each submission

    #     elif action == 'reset':
    #         chat_history = []  # Reset chat history
    #         queryNum = 0  # Reset queryNum to 0

@app.route('/upload', methods=['POST'])
def upload():
    # Check if the POST request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
 
    # Check if the file has a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Check if the file is an allowed format (e.g., Excel)
    allowed_extensions = {'xls', 'xlsx'}
    if not '.' in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': 'Unsupported file format'})

    # saving the file to disk
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))

    return jsonify({'message': 'File uploaded successfully'})

@app.route('/delete_files', methods=['POST'])
def delete_files():
    # Delete all files in the uploads folder
    file_list = os.listdir(app.config['UPLOAD_FOLDER'])
    for file_name in file_list:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        os.remove(file_path)
    return jsonify({'message': 'Uploaded files deleted successfully'})

@app.route('/get-table')
def get_table():
    
    print('trying to get table')
    print('does this change')
    df = get_display_df()
    print(df)
    # Convert DataFrame to HTML
    html = df.to_html(classes='dataframe', border=2)
    display = False
    # Return the HTML in a JSON response
    return jsonify({'html': html})


# Dummy database for demonstration purposes
users = {}

@app.route('/login', methods=['POST'])
def login():
    global signed_in
    data = request.get_json()  # Parse JSON data from request body
    print(data)
    username = data['loginUsername']
    password = data['loginPassword']
    print(f"Attempting login for {username}.")  # Log attempt
    user = users.get(username)
    if check_credentials(username, password):
        print(f"Login successful for {username}.")  # Log success
        signed_in = True
        return jsonify({'message': 'Login successful'}), 200
    else:
        print(f"Login failed for {username}.")  # Log failure
        return jsonify({'message': 'Invalid username or password'}), 401

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()  # Parse JSON data from request body
    print(data)
    username = data['registerUsername']
    password = data['registerPassword']
    print(f"Attempting registration for {username}.")  # Log attempt
    if add_user(username, password):
        print(f"User registered successfully: {username}.")  # Log success
        return jsonify({'message': 'User registered successfully'}), 201
    else:
        print(f"Registration failed for {username}: Username already exists.")  # Log failure
        return jsonify({'message': 'Username already exists'}), 400

@app.route('/get-array')
def get_array():
    return jsonify(get_prev_queries())


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000)


