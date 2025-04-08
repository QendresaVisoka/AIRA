import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';

const AnalyzedImage = ({ fileName, originalImage, maskImage }) => {
  const navigate = useNavigate();

  const [showBoxes, setShowBoxes] = useState(true);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [boxes, setBoxes] = useState([]);
  const [overlayImage, setOverlayImage] = useState(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const canvasRef = useRef(null);

  // Fetch bounding boxes
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

  // Fetch overlay image
  useEffect(() => {
    const fetchOverlayImage = async () => {
      try {
        const res = await fetch('http://localhost:5000/get-overlay?' + new Date().getTime());
        if (res.ok) {
          const blob = await res.blob();
          const overlayImageUrl = URL.createObjectURL(blob);
          setOverlayImage(overlayImageUrl);
        } else {
          alert('No overlay image found');
        }
      } catch (err) {
        alert('Failed to load overlay image: ' + err.message);
      }
    };
    fetchOverlayImage();
  }, []);

  // Draw bounding boxes
  const drawBoxes = useCallback(() => {
    const canvas = canvasRef.current;
    const img = document.getElementById('analyzed-image');

    if (!canvas || !img || !imageLoaded) return;

    const ctx = canvas.getContext('2d');
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

      {/* Flex row: images + toggles */}
      <div style={{ display: 'flex', gap: '40px', alignItems: 'flex-start' }}>
        {/* Image pair */}
        <div className="image-pair">
          <div style={{ maxWidth: '48%' }}>
            <img
              src={originalImage}
              alt="Original"
              className="img-original"
              style={{ width: '100%', display: 'block' }}
            />
          </div>

          <div style={{ position: 'relative', maxWidth: '48%' }}>
            <img
              id="analyzed-image"
              src={showHeatmap && overlayImage ? overlayImage : maskImage}
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

        {/* Toggle controls (aligned right of images) */}
        <div className="toggle-controls">
          <div className="custom-toggle" style={{ marginBottom: '20px' }}>
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

          <div className="custom-toggle">
            <label className="switch">
              <input
                type="checkbox"
                checked={showHeatmap}
                onChange={() => setShowHeatmap(prev => !prev)}
              />
              <span className="slider"></span>
            </label>
            <span style={{ marginLeft: '12px' }}>
              {showHeatmap ? 'Hide Heatmap' : 'Show Heatmap'}
            </span>
          </div>
        </div>
      </div>

      {/* Buttons */}
      <div style={{ marginTop: '30px' }}>
        <button className="top-left-button" onClick={() => navigate('/image')}>
          Back
        </button>
        <button className="bottom-right-button" onClick={() => navigate('/')}>
          Upload New File
        </button>
      </div>
    </div>
  );
};

export default AnalyzedImage;
