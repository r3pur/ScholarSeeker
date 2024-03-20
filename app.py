from flask import Flask
from views import views
from main import *

app = Flask(__name__)

app.register_blueprint(views, url_prefix="/")

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000)