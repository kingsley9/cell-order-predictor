// footer.tsx
import React from "react";
import { Link } from "react-router-dom";
import "./Footer.css";

const Footer = () => {
  return (
    <footer className="footer">
      <div className="footer-text">
        © 2023 Cell Order Predictor. All rights reserved.
      </div>
      <div className="footer-links">
        <Link className="footer-link" to="/about">
          About
        </Link>
        <Link className="footer-link" to="/termsOfService">
          Terms of Service
        </Link>
      </div>
    </footer>
  );
};

export default Footer;
