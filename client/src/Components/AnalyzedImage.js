import React from 'react';

const AnalyzedImage = ({ preprocessedImage, maskImage }) => {
  return (
    <div className="image-container">
        <div>
          <img src={maskImage} alt="Mask" />
        </div>
    </div>
  );
};

export default AnalyzedImage;