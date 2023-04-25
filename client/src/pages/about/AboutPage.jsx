import React from 'react';

import './AboutPage.css';

const AboutPage = () => {
  return (
    <div className="about">
      <h1 style={{ fontSize: '60px' }}>About Us</h1>
      <div className="container">
        <p>
          NotePredict is an innovative machine learning-driven application
          designed to streamline the organization of Jupyter notebooks by
          predicting the optimal order of markdown cells.
        </p>
        <p>
          The primary goal of this project is to save time and reduce
          frustration for users, such as data scientists, researchers, and
          analysts, who frequently work with Jupyter notebooks. NotePredict's
          user-friendly interface enables seamless uploading of notebooks and
          visualization of the predicted cell order, allowing users to focus on
          their research and analysis tasks.
        </p>
        <p>
          NotePredict is a powerful tool for software developers of all levels,
          from beginners to experienced professionals. By providing quick and
          accurate code readability analysis, NotePredict can help developers
          save time and increase productivity, allowing them to focus on what
          they do best: solving complex problems and creating innovative
          software solutions.
        </p>
      </div>
    </div>
  );
};

export default AboutPage;
