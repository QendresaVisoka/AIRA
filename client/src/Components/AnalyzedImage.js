import React from 'react';

const AnalyzedImage = ({ fileName, maskImage }) => {
  return (
    <div className="image-container">
      <h3>{fileName}</h3>
          <img src={maskImage} alt="Mask" />
    </div>
  );
};

export default AnalyzedImage;
