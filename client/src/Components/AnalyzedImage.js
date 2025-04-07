import React from 'react';
import { useNavigate } from 'react-router-dom';

const AnalyzedImage = ({ fileName, originalImage, maskImage }) => {
  const navigate = useNavigate();

  return (
    <div className="image-container">
      <h3>{fileName}</h3>
      <div className="image-pair">
        <img src={originalImage} alt="Original"/>
        <img src={maskImage} alt="Mask" />
      </div>

      <button className="top-left-button" onClick={() => navigate('/image')}>
        Back
      </button>

      <button className="button" onClick={() => navigate('/')}>
        Upload New File
      </button>
    </div>
  );
};

export default AnalyzedImage;
