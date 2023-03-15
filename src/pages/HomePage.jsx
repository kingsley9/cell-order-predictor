import React from 'react';
import { Button, Form, FormGroup, Label, Input } from 'reactstrap';
import './HomePage.css';

const HomePage = () => {
  const handleFileSelect = (event) => {
    const files = event.target.files;
    console.log(files);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    console.log('Form submitted');
  };

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
