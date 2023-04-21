import React, { useState, useEffect } from 'react';
import { Button } from 'reactstrap';
import { API_URL } from '../../api';
import { useParams } from 'react-router-dom';
import './DownloadPage.css';
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import 'react-tabs/style/react-tabs.css';
import { css } from '@emotion/react';
import { ClipLoader } from 'react-spinners';
const DownloadPage = () => {
  const { notebookId } = useParams();
  const [htmlOutput, setHtmlOutput] = useState('');
  const [loading, setLoading] = useState(true);
  const [notebookName, setNotebookName] = useState('');
  const [notebookData, setNotebookData] = useState('');
  const [readabilityScore, setReadabilityScore] = useState(0);
  const [originalHtmlOutput, setOriginalHtmlOutput] = useState('');
  const [notebookWidth] = useState('80%');

  useEffect(() => {
    fetch(`${API_URL}/download?notebook_id=${notebookId}`)
      .then((response) => {
        if (response.ok) {
          return response.json();
        }
        throw new Error('Network response was not ok.');
      })
      .then((data) => {
        setNotebookName(`generated_notebook_${notebookId}.ipynb`);
        setNotebookData(data.notebook);
        setHtmlOutput(data.html_output);
        setOriginalHtmlOutput(data.original_html_output);
        setReadabilityScore(data.readability_score);
      })
      .catch((error) => {
        console.error(error);
      })
      .finally(() => {
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
        <ClipLoader
          css={css`
            display: inline-block;
            margin-left: 10px;
          `}
          size={20}
          color={'#665894'}
          loading={loading}
        />
      ) : notebookData ? (
        <div style={{ flexDirection: 'column' }}>
          <Button className="download-button" onClick={downloadNotebook}>
            Download sorted Notebook
          </Button>
          <div className="readability-score">
            <p>Readability Score: {readabilityScore}%</p>
          </div>
          <div className="centered-box">
            <Tabs defaultIndex={1}>
              <TabList>
                <Tab>Original Notebook</Tab>
                <Tab>Sorted Notebook</Tab>
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
        </div>
      ) : (
        <div className="centered-box">
          <p>No notebook data available.</p>
        </div>
      )}
    </div>
  );
};

export default DownloadPage;
