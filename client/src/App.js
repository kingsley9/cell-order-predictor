import React from "react";
import HomePage from "./pages/home/HomePage";
import DownloadPage from "./pages/download/DownloadPage";
import AboutPage from "./pages/about/AboutPage";
import TermsOfServicePage from "./pages/termsOfService/TermsOfService";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import "./App.css";
import Header from "./components/shared/Header";
import Footer from "./components/shared/Footer";

function App() {
  return (
    <Router>
      <Header />
      <div className="content">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/download/:notebookId" element={<DownloadPage />} />
          <Route path="/termsOfService" element={<TermsOfServicePage />} />
        </Routes>
      </div>
      <Footer />
    </Router>
  );
}

export default App;
