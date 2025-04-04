import React, { useState } from 'react';
import './App.css';
import { BrowserRouter as Router, Route, Routes, useNavigate } from 'react-router-dom';
import UploadDICOM from './Components/UploadDICOM';
import AnalyzedImage from './Components/AnalyzedImage';

function App() {
  const [dicomImage, setDicomImage] = useState(null);
  const [preprocessedImage, setPreprocessedImage] = useState(null);
  const [maskImage, setMaskImage] = useState(null);

  return (
    <Router>
      <AppContent
        dicomImage={dicomImage}
        setDicomImage={setDicomImage}
        preprocessedImage={preprocessedImage}
        setPreprocessedImage={setPreprocessedImage}
        maskImage={maskImage}
        setMaskImage={setMaskImage}
      />
    </Router>
  );
}

function AppContent({
  dicomImage,
  setDicomImage,
  preprocessedImage,
  setPreprocessedImage,
  maskImage,
  setMaskImage,
}) {
  const navigate = useNavigate();

  const handleAnalyze = async () => {
    try {
      // Step 1: Preprocess
      const preResponse = await fetch('http://localhost:5000/preprocess-dicom', {
        method: 'POST',
      });

      if (!preResponse.ok) throw new Error('Preprocessing failed');

      const preBlob = await preResponse.blob();
      const preUrl = URL.createObjectURL(preBlob);
      setPreprocessedImage(preUrl);

      // Step 2: Predict
      const predResponse = await fetch('http://localhost:5000/predict-mask', {
        method: 'POST',
      });

      if (!predResponse.ok) throw new Error('Prediction failed');

      const predBlob = await predResponse.blob();
      const predUrl = URL.createObjectURL(predBlob);
      setMaskImage(predUrl);

      // Step 3: Navigate to analyzed-image
      navigate('/analyzed-image');
    } catch (err) {
      alert('Analysis failed: ' + err.message);
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
                navigateToImage={() => navigate('/image')}
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
                    <button className="button" onClick={handleAnalyze}>
                      Analyze
                    </button>
                    <button className="button" onClick={() => navigate('/')}>
                      Upload New File
                    </button>
                  </div>
                ) : (
                  <p>No image uploaded.</p>
                )}
              </div>
            }
          />

          <Route
            path="/analyzed-image"
            element={
              <AnalyzedImage
                preprocessedImage={preprocessedImage}
                maskImage={maskImage}
              />
            }
          />
        </Routes>
      </div>
    </div>
  );
}

export default App;
