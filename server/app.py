import tempfile
import requests
from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS
import nbformat
import uuid
import json


app = Flask(__name__)
api = Api(app)

CORS(app)  # enable CORS for all routes

app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()


def notebook_to_json(path):
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
    cells = {}
    sources = {}
    for i, cell in enumerate(nb.cells):
        cell_type = cell.cell_type

        # Generate a unique ID
        cell_id = str(uuid.uuid4())[:8]
        cells[cell_id] = cell_type
        sources[cell_id] = cell.source
    root = {
        "root": {
            "cell_type": cells,
            "source": sources
        }
    }
    return root


class UploadResource(Resource):
    # POST method
    def post(self):
        file = request.files['file']

        with tempfile.NamedTemporaryFile(delete=False) as temp:
            filepath = temp.name
            file.save(filepath)
            json_data = notebook_to_json(filepath)
        filename = uuid.uuid4().hex[:14] + '.json'
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=4)
        # Example: send the file to a Kaggle notebook
        url = 'https://www.kaggle.com/notebook/your-notebook-url/run'
        data = {'args': f'-f {file.filename}'}
        files = {'file': file.read()}
        # response = requests.post(url, data=data, files=files)
        print('File name:', file.filename)
        print('JSON:', filename)
        return json_data


api.add_resource(UploadResource, "/upload")

if __name__ == "__main__":
    app.run()
