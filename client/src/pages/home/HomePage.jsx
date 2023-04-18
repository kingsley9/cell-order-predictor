import React, { useState } from 'react';
import { Button, Form, FormGroup, Label, Input } from 'reactstrap';
import './HomePage.css';
import { API_URL } from '../../api';

const HomePage = () => {
  const [file, setFile] = useState(null); // Initialize the state with a null value

  function handleFileSelect(event) {
    const selectedFile = event.target.files[0];
    console.log(selectedFile?.name); // Print the name of the selected file to the console
    setFile(selectedFile); // Update the state with the selected file
  }

  function handleSubmit(event) {
    event.preventDefault();

    if (!file) {
      console.log('No file selected');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    fetch(`${API_URL}/upload`, {
      method: 'POST',
      body: formData,
    })
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        console.log(data); // Log the response from the server
        console.log(data.predictions);
        // Handle the response as needed
      })
      .catch((error) => {
        console.error(error); // Log any errors that occur
        // Handle the error as needed
      });
  }

  return (
    <div className="home-container rounded border p-4">
      <div className="home-upload-container">
        <Form onSubmit={handleSubmit}>
          <FormGroup>
            <Label className="file-upload-label">
              Please Upload Your Notebook
            </Label>
            <div className="file-upload-container">
              <Input
                type="file"
                name="file"
                id="fileUpload"
                accept=".ipynb"
                className="file-upload-input"
                onChange={handleFileSelect}
              />
              <Button className="file-upload-btn">Sort cells</Button>
            </div>
          </FormGroup>
        </Form>
      </div>
    </div>
  );
};

export default HomePage;
