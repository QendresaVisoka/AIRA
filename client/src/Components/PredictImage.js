import React from 'react';

const PredictedImage = ({ preprocessedImage, maskImage }) => {
  return (
    <div className="predict-image-container">
      <h2>Prediction Results</h2>
      <div style={{ display: 'flex', justifyContent: 'center', gap: '3rem' }}>
        <div>
          <h3>Preprocessed Image</h3>
          <img
            src={preprocessedImage}
            alt="Preprocessed"
            style={{ maxWidth: '400px', border: '1px solid gray' }}
          />
        </div>
        <div>
          <h3>Predicted Mask</h3>
          <img
            src={maskImage}
            alt="Prediction Mask"
            style={{ maxWidth: '400px', border: '1px solid gray' }}
          />
        </div>
      </div>
    </div>
  );
};

export default PredictedImage;
