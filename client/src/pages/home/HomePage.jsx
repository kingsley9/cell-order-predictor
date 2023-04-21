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

  function handleFileSelect(event) {
    const selectedFile = event.target.files[0];
    console.log(selectedFile?.name);
    setFile(selectedFile);
    const formData = new FormData();
    formData.append('file', selectedFile);
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

  return (
    <div className="home-container rounded border p-4">
      <h1 className="file-upload-label">
        <span>
          Upload Notebook to Reorder Cells <br></br> (Readability Analysis)
        </span>
      </h1>
      <div className="home-upload-container">
        {loading ? (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              marginBottom: 'auto',
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

            <Label htmlFor="fileUpload" className="custom-upload-button">
              Upload Notebook
            </Label>
            <Input
              type="file"
              name="file"
              id="fileUpload"
              accept=".ipynb"
              className="file-upload-input"
              onChange={handleFileSelect}
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
          <a href="#" class="suggestion-example-notebook">
            <img src="https://cdn1.iconfinder.com/data/icons/file-format-set/64/2878-512.png" alt="Example notebook" className="rounded-example"></img>
          </a>
          <a href="#" class="suggestion-example-notebook">
            <img src="https://cdn1.iconfinder.com/data/icons/file-format-set/64/2878-512.png" alt="Example notebook" className="rounded-example"></img>
          </a>
          <a href="#" class="suggestion-example-notebook">
            <img src="https://cdn1.iconfinder.com/data/icons/file-format-set/64/2878-512.png" alt="Example notebook" className="rounded-example"></img>
          </a>
          <a href="#" class="suggestion-example-notebook">
            <img src="https://cdn1.iconfinder.com/data/icons/file-format-set/64/2878-512.png" alt="Example notebook" className="rounded-example"></img>
          </a>
        </div>

     
    </div>
  );
};

export default HomePage;
