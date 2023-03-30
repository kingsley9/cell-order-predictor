import requests
from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS

app = Flask(__name__)
api = Api(app)

CORS(app)  # enable CORS for all routes


class UploadResource(Resource):
    # POST method
    def post(self):
        file = request.files['file']
        # Process the file data here

        # Example: send the file to a Kaggle notebook
        url = 'https://www.kaggle.com/notebook/your-notebook-url/run'
        data = {'args': f'-f {file.filename}'}
        files = {'file': file.read()}
        # response = requests.post(url, data=data, files=files)
        print('File name:', file.filename)
        return file.filename


api.add_resource(UploadResource, "/upload")

if __name__ == "__main__":
    app.run()
