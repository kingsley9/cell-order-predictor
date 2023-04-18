import React from 'react';
import { Link } from 'react-router-dom';
import './Header.css';

const Header = () => {
  return (
    <div className="header">
      <div className="logo">
        GROUP<span style={{ color: '#a02f06' }}> 16</span>
      </div>
      <h1 className="title">
        <span style={{ color: '#a02f06' }}> Notebook</span> Cell Order Predictor
      </h1>
      <div className="menu">
        <Link to="/" className="menu-item">
          Home
        </Link>
        <Link to="/about" className="menu-item">
          About
        </Link>
      </div>
    </div>
  );
};

export default Header;
