import React from 'react';

const AnalyzedImage = ({ preprocessedImage, maskImage }) => {
  return (
    <div className="image-container">
      <h2>Analysis Result</h2>
      <img src={maskImage} alt="Predicted Mask" />
    </div>
  );
};

export default AnalyzedImage;
