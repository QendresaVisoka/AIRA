import React from 'react';
import { useNavigate } from 'react-router-dom';

const PreprocessedImage = ({ preprocessedImage }) => {
  const navigate = useNavigate();

  return (
    <div className="preprocessed-image-container">
      <img
        src={preprocessedImage}
        alt="Preprocessed DICOM"
        className="dicom-img"
      />
      <br />
      <button className="button" onClick={() => navigate('/image')}>
        Back to Original Image
      </button>
    </div>
  );
};

export default PreprocessedImage;
