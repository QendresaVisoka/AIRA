import React, { useState } from 'react';
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
  const [dicomData, setDicomData] = useState({ imageUrl: null, fileName: null, patientInfo: null });
  const [preprocessedImage, setPreprocessedImage] = useState(null);
  const [maskImage, setMaskImage] = useState(null);
  const [tumorFound, setTumorFound] = useState(null);

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
        setMaskImage(URL.createObjectURL(blob));
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

          {/* ✅ Show this only on the / route */}
          {location.pathname === '/' && (
            <button className="login-top-right" onClick={() => navigate('/login')}>
              Login
            </button>
          )}
        </header>

      

      <div className="main-content">
        <Routes>
          <Route path="/" element={<WelcomePage />} />

          <Route path="/login" element={<LoginPage />} />

          <Route
            path="/upload"
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
                      <div>
                        <img src={dicomData.imageUrl} alt="DICOM" />
                      </div>

                      <div className="patient-info-box">
                        <h4><strong>Patient Info:</strong></h4>
                        <div style={{ textAlign: "left" }}>
                          <p><strong>ID:</strong> {dicomData.patientInfo?.id || 'N/A'}</p>
                          <p><strong>Sex:</strong> {dicomData.patientInfo?.sex || 'N/A'}</p>
                          <p><strong>Age:</strong> {dicomData.patientInfo?.age || 'N/A'}</p>
                        </div>
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
