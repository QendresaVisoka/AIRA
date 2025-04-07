import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const AnalyzedImage = ({ fileName, originalImage }) => {
  const navigate = useNavigate();

  const [showBoxes, setShowBoxes] = useState(true); // State to control toggle
  const [maskImage, setMaskImage] = useState(null);

  const fetchMaskImage = async (withBoxes) => {
    try {
      const response = await fetch(`http://localhost:5000/predict-mask?show_boxes=${withBoxes}`, {
        method: 'POST'
      });
      if (!response.ok) throw new Error("Failed to fetch image");
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setMaskImage(url);
    } catch (err) {
      alert("Failed to load mask image: " + err.message);
    }
  };

  useEffect(() => {
    fetchMaskImage(showBoxes);
  }, [showBoxes]);

  const toggleBoxes = () => {
    setShowBoxes(prev => !prev);
  };

  return (
    <div className="image-container">
      <h3>{fileName}</h3>
      <div className="image-pair">
        <img src={originalImage} alt="Original" />
        <img src={maskImage} alt="Analyzed" />
      </div>

      <button className="top-left-button" onClick={() => navigate('/image')}>Back</button>
      <button className="button" onClick={() => navigate('/')}>Upload New File</button>

      <button
        className="button"
        style={{ marginTop: '20px' }}
        onClick={toggleBoxes}
      >
        {showBoxes ? 'Hide Bounding Boxes' : 'Show Bounding Boxes'}
      </button>
    </div>
  );
};

export default AnalyzedImage;
