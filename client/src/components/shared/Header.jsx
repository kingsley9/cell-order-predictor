import React from 'react';
import { Link } from 'react-router-dom';
import './Header.css';

const Header = () => {
  return (
    <div className="header">
      <div className="logo">
        <a href="/" style={{ textDecoration: 'none' }}>
          <h1 style={{ color: 'black' }}>
            <span style={{ color: '#e35420' }}>Note</span>
            Predict
          </h1>
        </a>
      </div>

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
