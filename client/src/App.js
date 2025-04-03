import React, { useState } from 'react';
import './App.css';
import { BrowserRouter as Router, Route, Routes, useNavigate } from 'react-router-dom';
import UploadDICOM from './Components/UploadDICOM';
import PreprocessedImage from './Components/PreprocessedImage';

function App() {
  const [dicomImage, setDicomImage] = useState(null);
  const [preprocessedImage, setPreprocessedImage] = useState(null);

  return (
    <Router>
      <AppContent
        dicomImage={dicomImage}
        setDicomImage={setDicomImage}
        preprocessedImage={preprocessedImage}
        setPreprocessedImage={setPreprocessedImage}
      />
    </Router>
  );
}

function AppContent({ dicomImage, setDicomImage, preprocessedImage, setPreprocessedImage }) {
  const navigate = useNavigate();

  const handlePreprocess = async () => {
    try {
      const response = await fetch('http://localhost:5000/preprocess-dicom', {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error('Failed to preprocess image');
      }

      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      setPreprocessedImage(imageUrl);
      navigate('/preprocessed-image');
    } catch (err) {
      alert('Error preprocessing image: ' + err.message);
    }
  };

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
          <Route
            path="/"
            element={
              <UploadDICOM
                setDicomImage={setDicomImage}
                setPreprocessedImage={setPreprocessedImage}
              />
            }
          />

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
                    <button className="button" onClick={handlePreprocess}>
                      Preprocess Image
                    </button>
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

          <Route
            path="/preprocessed-image"
            element={<PreprocessedImage preprocessedImage={preprocessedImage} />}
          />
        </Routes>
      </div>
    </div>
  );
}

export default App;
