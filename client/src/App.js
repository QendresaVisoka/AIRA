import React, { useState, useEffect } from 'react';
import './App.css';
import {
  BrowserRouter as Router,
  Route,
  Routes,
  useNavigate,
  useLocation
} from 'react-router-dom';

import LoginPage from './Components/LoginPage';
import UploadDICOM from './Components/UploadDICOM';
import AnalyzedImage from './Components/AnalyzedImage';
import WelcomePage from './Components/WelcomePage';

function App() {
  // Load only lightweight data from localStorage (no imageUrl)
  const [dicomData, setDicomData] = useState(() => {
    const saved = localStorage.getItem('dicomData');
    return saved
      ? { ...JSON.parse(saved), imageUrl: null } // We'll fill imageUrl after upload
      : { imageUrl: null, fileName: null, patientInfo: null };
  });

  const [maskImage, setMaskImage] = useState(null); // not persisted anymore
  const [tumorFound, setTumorFound] = useState(null);
  const [preprocessedImage, setPreprocessedImage] = useState(null); // not persisted

  // Save only lightweight info
  useEffect(() => {
    const { fileName, patientInfo } = dicomData;
    try {
      localStorage.setItem('dicomData', JSON.stringify({ fileName, patientInfo }));
    } catch (err) {
      console.warn('Could not store dicomData:', err.message);
    }
  }, [dicomData]);

  return (
    <Router>
      <AppContent
        dicomData={dicomData}
        setDicomData={setDicomData}
        preprocessedImage={preprocessedImage}
        setPreprocessedImage={setPreprocessedImage}
        maskImage={maskImage}
        setMaskImage={setMaskImage}
        tumorFound={tumorFound}
        setTumorFound={setTumorFound}
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
  tumorFound,
  setTumorFound,
}) {
  const navigate = useNavigate();
  const location = useLocation();
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

      const result = await predResponse.json();
      setTumorFound(result.tumorFound ?? false);

      if (result.image) {
        const base64Data = result.image.split(',')[1];
        const byteCharacters = atob(base64Data);
        const byteNumbers = new Array(byteCharacters.length)
          .fill()
          .map((_, i) => byteCharacters.charCodeAt(i));
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: 'image/png' });
        const url = URL.createObjectURL(blob);
        setMaskImage(url);
      }

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
          <h1 className="site-title">AIRA – AI Radiology Assistant for Mammography</h1>
        </div>

        {location.pathname === '/' && (
          <button className="login-top-right" onClick={() => navigate('/login')}>
            Login
          </button>
        )}
      </header>

      <div className="main-content">
        <Routes>
          <Route path="/" element={<WelcomePage />} />

          <Route 
            path="/login" 
            element={
            <LoginPage />} />

          <Route
            path="/upload"
            element={
              <UploadDICOM
                setDicomData={(data) => {
                  // Clear previous analysis results
                  setMaskImage(null);
                  setTumorFound(null);
                  localStorage.removeItem('dicomData');

                  // Set new data (with imageUrl now included)
                  setDicomData(data);
                }}
                setMaskImage={setMaskImage}
                setTumorFound={setTumorFound}
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
                      <div className="patient-info-box">
                        <h4><strong>Patient Info:</strong></h4>
                        <div style={{ textAlign: 'left' }}>
                          <p><strong>ID:</strong> {dicomData.patientInfo?.id || 'N/A'}</p>
                          <p><strong>Sex:</strong> {dicomData.patientInfo?.sex || 'N/A'}</p>
                          <p><strong>Age:</strong> {dicomData.patientInfo?.age || 'N/A'}</p>
                        </div>
                      </div>

                      <div className="image-wrapper">
                        <img src={dicomData.imageUrl} alt="DICOM" />
                      </div>
                    </div>

                    <div>
                      <button className="top-left-button" onClick={() => navigate('/upload')}>Back</button>

                      {loading ? (
                        <div className="loading-container">
                          <span className="loader"></span>
                          <p>Analyzing</p>
                        </div>
                      ) : (
                        <button className="button" onClick={handleAnalyze}>Analyze</button>
                      )}

                      {/* Show forward button only if analysis is complete */}
                      {maskImage && tumorFound !== null && (
                        <button className="top-right-button" onClick={() => navigate('/analyzed-image')}>Forward</button>
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
                dicomData={dicomData}
                fileName={dicomData.fileName}
                originalImage={dicomData.imageUrl}
                maskImage={maskImage}
                tumorFound={tumorFound}
              />
            }
          />
        </Routes>
      </div>
    </div>
  );
}

export default App;
