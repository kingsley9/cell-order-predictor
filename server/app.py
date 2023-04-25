import base64
import io
import os
import tempfile
from flask import Flask, Response, jsonify, request, send_from_directory, send_file
from flask_restful import Resource, Api
from flask_cors import CORS
import nbformat
import uuid
import json
import pandas as pd
import torch
# from md_cell_predictor import initialize_model, predict
from md_cell_predictor_v2 import predict_df
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert import HTMLExporter


app = Flask(__name__)
api = Api(app)

CORS(app)  # enable CORS for all routes

app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Model
# model = initialize_model()


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

    return df, nb  # Return the original notebook as well


def create_new_notebook(df_data, predictions):
    nb = new_notebook()
# Convert predictions from a Series to a list of cell IDs
    cell_ids = predictions.tolist()[0].split(' ')

    # Sort df_data based on the order of the predictions
    df_data = df_data.set_index('cell_id')

    df_data = df_data.loc[cell_ids]
    df_data = df_data.reset_index()
    for index, row in df_data.iterrows():
        cell_id = row.name
        cell_type = row['cell_type']
        source = row['source']

        if cell_type == "markdown":
            nb.cells.append(new_markdown_cell(source))
        elif cell_type == "code":
            nb.cells.append(new_code_cell(source))
    # print("DF:", df_data)

    # Save the new notebook to a BytesIO object
    notebook_data = io.BytesIO()
    notebook_data.write(nbformat.writes(nb).encode())
    notebook_data.seek(0)

    return notebook_data, nb


def calculate_readability_score(predictions, df_data):
    # Convert predictions from a Series to a list of cell IDs
    predicted_cell_ids = predictions.tolist()[0].split(' ')

    # Get the original order of cell IDs
    original_order = df_data['cell_id'].tolist()

    # print(f"original_order: {original_order}")
    # print(f"predicted_order: {predicted_cell_ids}")
    # Find the matching cell IDs
    matching_cells = [original_order[i] for i in range(
        len(original_order)) if original_order[i] == predicted_cell_ids[i]]

    # Count the number of correctly placed cells
    num_incorrectly_placed = sum([1 for i in range(
        len(original_order)) if original_order[i] != predicted_cell_ids[i]])
    num_correctly_placed = len(original_order) - num_incorrectly_placed/2
    # Calculate the readability score
    score = (num_correctly_placed / len(original_order)) * 100

    # Round the score to two decimal places
    score = round(score, 2)
    return score


cache = {}


class UploadResource(Resource):
    # POST method
    def post(self):
        file = request.files['file']

        with tempfile.NamedTemporaryFile(delete=False) as temp:
            filepath = temp.name
            file.save(filepath)
            notebook_id = uuid.uuid4().hex[:14]
            df_data, original_notebook = notebook_to_dataframe(
                filepath, notebook_id)  # Get the original notebook

        # print("df:", df_data)
        filename = uuid.uuid4().hex[:14] + '.json'

        # Create the output directory if it doesn't exist
        # output_dir = "./output/"
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)

        # # Write the JSON file to the output directory
        # with open(output_dir + filename, 'w') as f:
        #     json.dump(df_data.to_dict(), f, indent=4)
        # Make predictions using the pre-trained model and the provided data
        predictions = predict_df(df_data)

        cache[notebook_id] = {
            "df_data": df_data,
            "predictions": predictions,
            "original_notebook": original_notebook,
        }
        return {"predictions":  predictions.to_dict()[0], "notebook_id": notebook_id}


class DownloadResource(Resource):
    # GET method
    def get(self):
        notebook_id = request.args.get("notebook_id")
        if notebook_id in cache:
            df_data = cache[notebook_id]["df_data"]
            predictions = cache[notebook_id]["predictions"]
            original_notebook = cache[notebook_id]["original_notebook"]
            score = calculate_readability_score(predictions, df_data)
            # Create the HTML output for both original and new notebooks
            exporter = HTMLExporter()

            # Get both the BytesIO object and the NotebookNode object
            new_notebook_data, notebook_node = create_new_notebook(
                df_data, predictions)
            new_notebook_data.seek(0)
            (original_html_output, _) = exporter.from_notebook_node(
                original_notebook)  # Convert the original notebook to HTML
            # Create the HTML output
            (html_output, _) = exporter.from_notebook_node(notebook_node)

            # Convert the bytes object to a base64-encoded string
            notebook_base64 = base64.b64encode(
                new_notebook_data.getvalue()).decode('utf-8')

            response = jsonify({
                'notebook': notebook_base64,
                'html_output': html_output,
                'original_html_output': original_html_output,
                'readability_score': score,
            })

            return response
        else:
            return {"error": "Notebook not found"}, 404


api.add_resource(UploadResource, "/upload")
api.add_resource(DownloadResource, "/download")
if __name__ == "__main__":
    app.run()
