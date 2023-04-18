import tempfile
from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS
import nbformat
import uuid
import json
import pandas as pd


from md_cell_predictor import MarkdownModel, predict


app = Flask(__name__)
api = Api(app)

CORS(app)  # enable CORS for all routes

app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Model
model = MarkdownModel()


def notebook_to_dataframe(path, notebook_id):
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
    data = []
    for i, cell in enumerate(nb.cells):
        cell_type = cell.cell_type

        # Generate a unique ID
        cell_id = str(uuid.uuid4())[:8]
        source = cell.source
        data.append([notebook_id, cell_id, cell_type, source])
    df = pd.DataFrame(data, columns=['id', 'cell_id', 'cell_type', 'source'])

    # Add rank, pred, and pct_rank columns
    df["rank"] = df.groupby(["id", "cell_type"]).cumcount()
    df["pred"] = df.groupby(["id", "cell_type"])["rank"].rank(pct=True)
    df["pct_rank"] = 0

    return df


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
            notebook_id = uuid.uuid4().hex[:14]
            df_data = notebook_to_dataframe(filepath, notebook_id)
        # print("df:", df_data)
        filename = uuid.uuid4().hex[:14] + '.json'
        with open(filename, 'w') as f:
            json.dump(df_data.to_dict(), f, indent=4)
        # Make predictions using the pre-trained model and the provided data
        predictions = predict(model, df_data)
        print("preds:", predictions.to_dict()[0])

        # Return the DataFrame data along with the predictions
        return {"data": df_data.to_json(orient="records"), "predictions":  predictions.to_dict()[0]}


api.add_resource(UploadResource, "/upload")

if __name__ == "__main__":
    app.run()
