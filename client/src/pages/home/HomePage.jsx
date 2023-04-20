import React, { useState } from 'react';
import { Button, Form, FormGroup, Label, Input } from 'reactstrap';
import './HomePage.css';
import { API_URL } from '../../api';
import { css } from '@emotion/react';
import { ClipLoader } from 'react-spinners';
import { useNavigate } from 'react-router-dom';

const HomePage = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  function handleFileSelect(event) {
    const selectedFile = event.target.files[0];
    console.log(selectedFile?.name);
    setFile(selectedFile);
  }

  function handleSubmit(event) {
    event.preventDefault();

    if (!file) {
      console.log('No file selected');
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    fetch(`${API_URL}/upload`, {
      method: 'POST',
      body: formData,
    })
      .then((response) => {
        setLoading(false);
        return response.json();
      })
      .then((data) => {
        console.log(data);
        console.log(data.predictions);
        navigate(`/download/${data.notebook_id}`); // Navigate to downloads with notebookId as a parameter
      })
      .catch((error) => {
        console.error(error);
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
              <ClipLoader
                css={css`
                  display: inline-block;
                  margin-left: 10px;
                `}
                size={20}
                color={'#665894'}
                loading={loading}
              />
            </div>
          </FormGroup>
        </Form>
      </div>
    </div>
  );
};

export default HomePage;
