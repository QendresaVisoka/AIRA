import React from 'react';

const AnalyzedImage = ({ preprocessedImage, maskImage }) => {
  return (
    <div className="analyzed-image-container">
      <h2>Analysis Result</h2>
      <div style={{ display: 'flex', justifyContent: 'center', gap: '2rem' }}>
        <div>
          <h3>Preprocessed Image</h3>
          <img src={preprocessedImage} alt="Preprocessed" style={{ maxWidth: '400px' }} />
        </div>
        <div>
          <h3>Predicted Mask</h3>
          <img src={maskImage} alt="Mask" style={{ maxWidth: '400px' }} />
        </div>
      </div>
    </div>
  );
};

export default AnalyzedImage;
