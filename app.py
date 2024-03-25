from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from werkzeug.utils import secure_filename
from main import *


app = Flask(__name__)

# Initialize global variables
chat_history = []
queryNum = 0  

# # Define the directory where uploaded files will be saved
# UPLOAD_FOLDER = 'uploads'  # Replace this with the actual path on your server
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    global chat_history, queryNum  # Refer to the global variables
    user_query = request.json['message']
    num_input = int(request.form.get('num', 2))  # Default to 1 if not provided
    response = user_question(user_query, num_input, queryNum=queryNum)
    chat_history = chat_history[-5:]  # Keep last 6 messages
    chat_history += [{'user': user_query}, {'scholarseeker': response}]
    queryNum += 1  # Increment queryNum by 1 for each submission
    chat_history = []  

    return jsonify({'message': response})

@app.route('/reset', methods=['POST'])
def reset():
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000)