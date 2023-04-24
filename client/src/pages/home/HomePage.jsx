import React, { useState } from 'react';
import { FormGroup, Label, Input } from 'reactstrap';
import './HomePage.css';
import { API_URL } from '../../api';
import { css } from '@emotion/react';
import { ClipLoader } from 'react-spinners';
import { useNavigate } from 'react-router-dom';

const HomePage = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  function handleFileSelect(selectedFile) {
    let fileToUpload;

    if (typeof selectedFile === 'string') {
      // Read the file at the specified path and create a File object from it
      fetch(selectedFile)
        .then((response) => {
          if (!response.ok) {
            throw new Error(`Failed to fetch file: ${response.status}`);
          }
          return response.blob();
        })
        .then((blob) => {
          fileToUpload = new File([blob], selectedFile.split('/').pop());
          console.log(fileToUpload?.name);
          console.log(fileToUpload);

          submitFile(fileToUpload);
        })
        .catch((error) => {
          console.error(error);
        });
    } else {
      fileToUpload = selectedFile;
      console.log(fileToUpload?.name);

      submitFile(fileToUpload);
    }

    function submitFile(file) {
      setFile(file);

      const formData = new FormData();
      formData.append('file', file);

      setLoading(true);

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
          navigate(`/download/${data.notebook_id}`);
        })
        .catch((error) => {
          console.error(error);
        });
    }
  }
  function handleDrop(event) {
    event.preventDefault();
    const droppedFile = event.dataTransfer.files[0];
    handleFileSelect(droppedFile);
  }

  function handleDragOver(event) {
    event.preventDefault();
  }

  return (
    <div className="home-container">
      <h1 className="file-upload-label">
        <span>
          Effortlessly Arrange Your Python Notebook <br></br> with AI-Powered
          Cell Ordering
        </span>
      </h1>
      <div
        className="home-upload-container"
        onDragEnter={(e) => e.preventDefault()}
        onDragLeave={(e) => e.preventDefault()}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        {loading ? (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              margin: 'auto',
            }}
          >
            <ClipLoader
              css={css`
                display: inline-block;
              `}
              size={20}
              color={'#665894'}
              loading={loading}
            />
            <p style={{ marginLeft: '10px' }}>Sorting Notebook</p>
          </div>
        ) : (
          <FormGroup>
            <img
              src="https://cdn-icons-png.flaticon.com/512/126/126477.png"
              className="notebook-file"
            ></img>
            <br></br>
            <Label htmlFor="fileUpload" className="custom-upload-button">
              Upload Notebook
            </Label>
            <br></br>
            <h3 className="drag-drop"> Or drag and drop </h3>
            <Input
              type="file"
              name="file"
              id="fileUpload"
              accept=".ipynb"
              className="file-upload-input"
              onChange={(e) => handleFileSelect(e.target.files[0])}
            />
          </FormGroup>
        )}
      </div>
      <div className="suggestion">
        <div className="suggestion-text">
          <div>No Notebook?</div>
          <div>Try one of our notebooks:</div>
        </div>
        <div className="suggestion-size">
          <a
            href="#"
            className="suggestion-example-notebook"
            onClick={() => handleFileSelect('/test.ipynb')}
          >
            <img
              src="https://cdn1.iconfinder.com/data/icons/file-format-set/64/2878-512.png"
              alt="Example notebook"
              className="rounded-example"
            ></img>
          </a>
          <a
            href="#"
            className="suggestion-example-notebook"
            onClick={() => handleFileSelect('/test2.ipynb')}
          >
            <img
              src="https://cdn1.iconfinder.com/data/icons/file-format-set/64/2878-512.png"
              alt="Example notebook"
              className="rounded-example"
            ></img>
          </a>
          <a
            href="#"
            className="suggestion-example-notebook"
            onClick={() => handleFileSelect('/test3.ipynb')}
          >
            <img
              src="https://cdn1.iconfinder.com/data/icons/file-format-set/64/2878-512.png"
              alt="Example notebook"
              className="rounded-example"
            ></img>
          </a>
          {/* <a href="#" class="suggestion-example-notebook">
            <img
              src="https://cdn1.iconfinder.com/data/icons/file-format-set/64/2878-512.png"
              alt="Example notebook"
              className="rounded-example"
            ></img>
          </a>
          <a href="#" class="suggestion-example-notebook">
            <img
              src="https://cdn1.iconfinder.com/data/icons/file-format-set/64/2878-512.png"
              alt="Example notebook"
              className="rounded-example"
            ></img>
          </a>
          <a href="#" class="suggestion-example-notebook">
            <img
              src="https://cdn1.iconfinder.com/data/icons/file-format-set/64/2878-512.png"
              alt="Example notebook"
              className="rounded-example"
            ></img>
          </a> */}
        </div>
      </div>
    </div>
  );
};

export default HomePage;
