from flask import Blueprint, render_template, request, redirect, url_for
from main import *

views = Blueprint(__name__, "views")

# Initialize global variables
chat_history = []
queryNum = 0  

@views.route("/", methods=['GET', 'POST'])
def search():
    global chat_history, queryNum  # Refer to the global variables
    if request.method == 'POST':
        action = request.form.get('action')  # Safely get the action value
        
        if action == 'submit':
            user_query = request.form.get('query', '')
            num_input = int(request.form.get('num', 1))  # Default to 1 if not provided
            response = user_question(user_query, num_input, queryNum=queryNum)
            chat_history = chat_history[-5:]  # Keep last 6 messages
            chat_history += [{'user': user_query}, {'scholarseeker': response}]
            queryNum += 1  # Increment queryNum by 1 for each submission

        elif action == 'reset':
            chat_history = []  # Reset chat history
            queryNum = 0  # Reset queryNum to 0
        
        return redirect(url_for('views.search'))
    
    return render_template('search.html', history=chat_history)