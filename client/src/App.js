import React, { useState } from 'react';
import './App.css';
import { BrowserRouter as Router, Route, Routes, useNavigate } from 'react-router-dom';
import UploadDICOM from './Components/UploadDICOM';

function App() {
  const [dicomImage, setDicomImage] = useState(null);

  return (
    <Router>
      <AppContent dicomImage={dicomImage} setDicomImage={setDicomImage} />
    </Router>
  );
}

// Extracted to access hooks like useNavigate
function AppContent({ dicomImage, setDicomImage }) {
  const navigate = useNavigate();

  return (
    <div className="App">
      <header className="App-header">
        <div className="header-left">
          <img src="/logo.png" alt="Logo" className="logo" />
          <h1 className="site-title">MammoCheck - Breast Cancer Detection</h1>
        </div>
      </header>

      <div className="main-content">
        <Routes>
          <Route path="/" element={<UploadDICOM setDicomImage={setDicomImage} />} />
          <Route
            path="/image"
            element={
              <div className="dicom-image-container">
                {dicomImage ? (
                  <div>
                    <img
                      src={dicomImage}
                      alt="DICOM"
                      style={{ width: '80%', height: 'auto', border: '1px solid black' }}
                    />
                    <br />
                    <button className="button" onClick={() => navigate('/')}>
                      Go Back and Upload a New File
                    </button>
                  </div>
                ) : (
                  <p>No image uploaded.</p>
                )}
              </div>
            }
          />
        </Routes>
      </div>
    </div>
  );
}

export default App;
