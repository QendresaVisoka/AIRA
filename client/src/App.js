import React, { useState } from 'react';
import './App.css';
import { BrowserRouter as Router, Route, Routes, useNavigate } from 'react-router-dom';
import UploadDICOM from './Components/UploadDICOM';
import AnalyzedImage from './Components/AnalyzedImage';


function App() {
  const [dicomData, setDicomData] = useState({ imageUrl: null, fileName: null });
  const [preprocessedImage, setPreprocessedImage] = useState(null);
  const [maskImage, setMaskImage] = useState(null);

  return (
    <Router>
      <AppContent
        dicomData={dicomData}
        setDicomData={setDicomData}
        preprocessedImage={preprocessedImage}
        setPreprocessedImage={setPreprocessedImage}
        maskImage={maskImage}
        setMaskImage={setMaskImage}
      />
    </Router>
  );
}

function AppContent({
  dicomData,
  setDicomData,
  preprocessedImage,
  setPreprocessedImage,
  maskImage,
  setMaskImage,
}) {
  const navigate = useNavigate();

  const handleAnalyze = async () => {
    try {
      const preResponse = await fetch('http://localhost:5000/preprocess-dicom', {
        method: 'POST',
      });

      if (!preResponse.ok) throw new Error('Preprocessing failed');

      const preBlob = await preResponse.blob();
      const preUrl = URL.createObjectURL(preBlob);
      setPreprocessedImage(preUrl);

      const predResponse = await fetch('http://localhost:5000/predict-mask', {
        method: 'POST',
      });

      if (!predResponse.ok) throw new Error('Prediction failed');

      const predBlob = await predResponse.blob();
      const predUrl = URL.createObjectURL(predBlob);
      setMaskImage(predUrl);

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
                setDicomData={setDicomData}
                navigateToImage={() => navigate('/image')}
              />
            }
          />

          <Route
            path="/image"
            element={
              <div className="image-container">
                {dicomData.imageUrl ? (
                  <div>
                    <h3>{dicomData.fileName}</h3>
                    <img src={dicomData.imageUrl} alt="DICOM" />
                    <br />
                    <button className="top-left-button" onClick={() => navigate('/')}>
                      Back
                    </button>
                    <button className="button" onClick={handleAnalyze}>
                      Analyze
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
                fileName={dicomData.fileName}
                originalImage={dicomData.imageUrl}
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