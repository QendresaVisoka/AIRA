import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';

const AnalyzedImage = ({ fileName, originalImage, maskImage }) => {
  const navigate = useNavigate();

  const [showBoxes, setShowBoxes] = useState(true);
  const [boxes, setBoxes] = useState([]);
  const [imageLoaded, setImageLoaded] = useState(false);
  const canvasRef = useRef(null);

  // Fetch bounding boxes (only once)
  useEffect(() => {
    const fetchBoundingBoxes = async () => {
      try {
        const res = await fetch('http://localhost:5000/get-bounding-boxes');
        const data = await res.json();
        if (data.boxes) setBoxes(data.boxes);
      } catch (err) {
        alert('Failed to load bounding boxes: ' + err.message);
      }
    };
    fetchBoundingBoxes();
  }, []);

  // Draw bounding boxes when toggled or image loads
  const drawBoxes = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageLoaded) return;

    const ctx = canvas.getContext('2d');
    const img = document.getElementById('analyzed-image');
    if (!img) return;

    // Set canvas size to match image
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (showBoxes && boxes.length > 0) {
      ctx.strokeStyle = 'yellow';
      ctx.lineWidth = 3;

      const xScale = img.clientWidth / img.naturalWidth;
      const yScale = img.clientHeight / img.naturalHeight;

      boxes.forEach(([x1, y1, x2, y2]) => {
        const sx = x1 * xScale;
        const sy = y1 * yScale;
        const ex = (x2 - x1) * xScale;
        const ey = (y2 - y1) * yScale;
        ctx.strokeRect(sx, sy, ex, ey);
      });
    }
  }, [showBoxes, boxes, imageLoaded]);

  useEffect(() => {
    drawBoxes();
  }, [drawBoxes]);

  return (
    <div className="image-container">
      <h3>{fileName}</h3>

      <div
        className="image-pair"
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '20px',
          justifyContent: 'center',
          alignItems: 'flex-start',
        }}
      >
        {/* Original Image on the Left */}
        <div style={{ maxWidth: '48%' }}>
          <img
            src={originalImage}
            alt="Original"
            className="img-original"
            style={{ width: '100%', display: 'block' }}
          />
        </div>

        {/* Analyzed Image with Canvas Overlay on the Right */}
        <div style={{ position: 'relative', maxWidth: '48%' }}>
          <img
            id="analyzed-image"
            src={maskImage}
            alt="Analyzed"
            onLoad={() => setImageLoaded(true)}
            style={{ width: '100%', display: 'block' }}
          />
          <canvas
            ref={canvasRef}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              pointerEvents: 'none',
              width: '100%',
              height: '100%',
            }}
          />
        </div>
      </div>

      <div style={{ marginTop: '30px' }}>
        <button className="top-left-button" onClick={() => navigate('/image')}>
          Back
        </button>
        <button className="button" onClick={() => navigate('/')}>
          Upload New File
        </button>
        <div className="custom-toggle" style={{ marginTop: '20px' }}>
          <label className="switch">
            <input
              type="checkbox"
              checked={showBoxes}
              onChange={() => setShowBoxes(prev => !prev)}
            />
            <span className="slider"></span>
          </label>
          <span style={{ marginLeft: '12px' }}>
            {showBoxes ? 'Hide Bounding Boxes' : 'Show Bounding Boxes'}
          </span>
        </div>
      </div>
    </div>
  );
}  

export default AnalyzedImage;