# Cell-Order-Predictor

This project is a cell order predictor for Jupyter notebooks that predicts the order of markdown cells using a pre-trained model. The system includes a Flask server that processes uploaded Jupyter notebooks and a React front-end that allows users to upload notebooks and view the predicted cell order.

## System Components

The main components of this system include:

- **Python Flask Server**: Handles file uploads and processes the input Jupyter notebooks, making predictions using the cell predictor script
- **Markdown Cell Predictor**: A Python script that includes the pre-trained and fine tuned model as well as necessary functions to predict the order of markdown cells.
- **React Front-end**: A user interface for uploading Jupyter notebooks and viewing the predicted cell order.

## Installation and Setup

### Flask Server

1. Create a virtual environment:
`python -m venv venv`
2. Activate the virtual environment:

- On Windows:
  ```
  venv\Scripts\activate
  ```
- On macOS/Linux:
  ```
  source venv/bin/activate
  ```
3. Install the required packages for the Flask server from the requirements.txt file:
`pip install -r requirements.txt`

4. Create a .env file in the root directory of your project to store your AWS access key, secret access key, and S3 bucket ARN. Make sure to add the .env file to your .gitignore to prevent it from being committed to version control.

Example .env file:
```
access_key=YOUR_AWS_ACCESS_KEY
secret_access_key=YOUR_AWS_SECRET_ACCESS_KEY
bucket_ARN=YOUR_S3_BUCKET_ARN
```

5. Run the Flask server:
`flask run`


### React Front-end

1. Install the required packages for the React front-end:
`npm i`

2. Create a file in the root directory of your React project to export the API URL.

Example api export:
`export const API_URL = 'http://localhost:5000';`

3. Run the React development server:
`npm start`


Your application should now be running, and you can access the React front-end at http://localhost:3000.
