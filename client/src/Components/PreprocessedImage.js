import React from 'react';
import { useNavigate } from 'react-router-dom';

const PreprocessedImage = ({ preprocessedImage, setMaskImage }) => {
  const navigate = useNavigate();

  const handlePredictMask = async () => {
    try {
      const response = await fetch('http://localhost:5000/predict-mask', {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error('Failed to get prediction.');
      }

      const blob = await response.blob();
      const maskUrl = URL.createObjectURL(blob);
      setMaskImage(maskUrl);
      navigate('/predicted-image');
    } catch (err) {
      alert('Error getting mask prediction: ' + err.message);
    }
  };

  return (
    <div className="preprocessed-image-container">
      <h2>Preprocessed Image</h2>
      <img
        src={preprocessedImage}
        alt="Preprocessed DICOM"
        className="dicom-img"
      />
      <br />
      <button className="button" onClick={handlePredictMask}>
      </button>
      <button className="button" onClick={() => navigate('/image')}>
        Back to Original Image
      </button>
    </div>
  );
};

export default PreprocessedImage;
