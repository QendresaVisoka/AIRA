import React, { useState } from 'react';
import './App.css';
import { BrowserRouter as Router, Route, Routes, useNavigate } from 'react-router-dom';
import UploadDICOM from './Components/UploadDICOM';
import AnalyzedImage from './Components/AnalyzedImage';

function App() {
  const [dicomData, setDicomData] = useState({ imageUrl: null, fileName: null, patientInfo: null });
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
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    setLoading(true);
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
    } finally {
      setLoading(false);
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
                  <div className="image-display">
                  <h3>{dicomData.fileName}</h3>
                
                  <div className="image-and-info-row">
                    {/* Image in the center */}
                    <div>
                      <img src={dicomData.imageUrl} alt="DICOM" />
                    </div>
                
                    {/* Patient info on the right */}
                    <div className="patient-info-box">
                      <h4><strong>Patient Info:</strong></h4>
                      <div style={{textAlign: "left"}}>
                        <p style={{margin: "5px"}}><strong>ID:</strong> {dicomData.patientInfo?.id || 'N/A'}</p>
                        <p style={{margin: "5px"}}><strong>Sex:</strong> {dicomData.patientInfo?.sex || 'N/A'}</p>
                        <p style={{margin: "5px"}}><strong>Age:</strong> {dicomData.patientInfo?.age || 'N/A'}</p>
                      </div>
                    </div>
                  </div>
                
                  {/* Analyze + Back Buttons */}
                  <div>
                    <button className="top-left-button" onClick={() => navigate('/')}>Back</button>
                    {loading ? (
                      <div className="loading-container">
                        <span className="loader"></span>
                        <p>Analyzing</p>
                      </div>
                    ) : (
                      <button className="button" onClick={handleAnalyze}>Analyze</button>
                    )}
                  </div>
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