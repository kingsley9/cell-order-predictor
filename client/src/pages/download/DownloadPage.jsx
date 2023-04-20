import React, { useState, useEffect, useRef } from 'react';
import { Button } from 'reactstrap';
import { API_URL } from '../../api';
import { useParams } from 'react-router-dom';
import './DownloadPage.css';
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs'; // Import Tabs components
import 'react-tabs/style/react-tabs.css'; // Import the default styling
const DownloadPage = () => {
  const { notebookId } = useParams(); // Retrieve notebook id from URL parameter
  const [htmlOutput, setHtmlOutput] = useState('');
  const [loading, setLoading] = useState(false);
  const [notebookName, setNotebookName] = useState('');
  const [notebookData, setNotebookData] = useState('');
  const [originalHtmlOutput, setOriginalHtmlOutput] = useState('');

  const [notebookWidth, setNotebookWidth] = useState('80%');

  useEffect(() => {
    // Fetch the generated notebook file from the server
    fetch(`${API_URL}/download?notebook_id=${notebookId}`)
      .then((response) => {
        if (response.ok) {
          return response.json();
        }
        throw new Error('Network response was not ok.');
      })
      .then((data) => {
        setLoading(false);
        setNotebookName(`generated_notebook_${notebookId}.ipynb`);
        setNotebookData(data.notebook);
        setHtmlOutput(data.html_output);
        setOriginalHtmlOutput(data.original_html_output);
      })
      .catch((error) => {
        console.error(error);
        setLoading(false);
      });
  }, [notebookId]);

  const downloadNotebook = () => {
    const blob = new Blob([atob(notebookData)], {
      type: 'application/octet-stream',
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = notebookName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="download-container">
      {loading ? (
        <p>Loading...</p>
      ) : (
        <div className="centered-box">
          <Tabs defaultIndex={1}>
            <TabList>
              <Tab>Original Notebook</Tab>
              <Tab>Sorted Notebook</Tab>

              <Button className="download-button" onClick={downloadNotebook}>
                Download Notebook
              </Button>
            </TabList>
            <TabPanel>
              <div
                className="html-output"
                dangerouslySetInnerHTML={{ __html: originalHtmlOutput }}
                style={{ width: notebookWidth }}
              ></div>
            </TabPanel>
            <TabPanel>
              <div
                className="html-output"
                dangerouslySetInnerHTML={{ __html: htmlOutput }}
                style={{ width: notebookWidth }}
              ></div>
            </TabPanel>
          </Tabs>
        </div>
      )}
    </div>
  );
};

export default DownloadPage;
