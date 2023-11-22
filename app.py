import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import json

from model import Model

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

print("Initializing model")
print("Training model")
md = Model()
print("Model ready. Starting app.")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return json.dumps({"error":"no file"}), 400
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return json.dumps({"error":"no file"}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # the file exists at <filename>
            return json.dumps(md.predict(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
    if request.method == 'GET':
        with open("html/index.html", "r") as f:
            return f.read()
    return "?"

