import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';

// State variables for image analysis
const AnalyzedImage = ({ fileName, originalImage, maskImage, tumorFound, dicomData }) => {
  const navigate = useNavigate();
  const [showBoxes, setShowBoxes] = useState(true);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [boxes, setBoxes] = useState([]);
  const [overlayImage, setOverlayImage] = useState(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const canvasRef = useRef(null);
  const [pixelSpacing, setPixelSpacing] = useState([1.0, 1.0]);
  const [legendUrl, setLegendUrl] = useState(null);

  // Fetch bounding boxes and pixel spacing from the server
  useEffect(() => {
    const fetchBoundingBoxes = async () => {
      try {
        const res = await fetch('http://localhost:5000/get-bounding-boxes');
        const data = await res.json();
        if (data.boxes) setBoxes(data.boxes);
        if (data.pixel_spacing) setPixelSpacing(data.pixel_spacing);
      } catch (err) {
        alert('Failed to load bounding boxes: ' + err.message);
      }
    };
    fetchBoundingBoxes();
  }, []);

  // Fetch overlay image from the server
  useEffect(() => {
    const fetchOverlayImage = async () => {
      try {
        const res = await fetch('http://localhost:5000/get-overlay?' + new Date().getTime());
        if (res.ok) {
          const blob = await res.blob();
          const overlayImageUrl = URL.createObjectURL(blob);
          setOverlayImage(overlayImageUrl);
        }
      } catch (err) {
        alert('Failed to load overlay image: ' + err.message);
      }
    };
    fetchOverlayImage();
  }, []);

  // Fetch heatmap legend from the server
  useEffect(() => {
    const fetchLegend = async () => {
      try {
        const res = await fetch('http://localhost:5000/get-heatmap-legend');
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        setLegendUrl(url);
      } catch (err) {
        console.error('Failed to load heatmap legend:', err);
      }
    };
    fetchLegend();
  }, []);

  // Draw bounding boxes on the canvas
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

  // Calculate tumor sizes in pixels and millimeters
  const getTumorSizes = () => {
    const [rowSpacing, colSpacing] = pixelSpacing;
    return boxes.map(([x1, y1, x2, y2], idx) => {
      const widthPx = x2 - x1;
      const heightPx = y2 - y1;
      const widthMM = (widthPx * colSpacing).toFixed(1);
      const heightMM = (heightPx * rowSpacing).toFixed(1);
      return {
        id: idx + 1,
        widthPx,
        heightPx,
        widthMM,
        heightMM
      };
    });
  };

  return (
    <div
      style={{
        backgroundImage: 'url("/login-background.png")',
        backgroundSize: 'cover',
        backgroundPosition: 'center',
       backgroundRepeat: 'no-repeat',
      width: '100vw',
      height: '100vh',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      position: 'relative',
      }}
    >
      <div className="image-container">
        <h3>{fileName}</h3>

        <div className="image-content-wrapper"  style={{ display: 'flex', gap: '40px', alignItems: 'flex-start' }}>
    
          <div className="patient-info-box" >
          <h4><strong>Patient Info:</strong></h4>
            <div style={{ textAlign: "left" }}>
              <p><strong>ID:</strong> {dicomData?.patientInfo?.id || 'N/A'}</p>
              <p><strong>Sex:</strong> {dicomData?.patientInfo?.sex || 'N/A'}</p>
              <p><strong>Age:</strong> {dicomData?.patientInfo?.age || 'N/A'}</p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '40px', alignItems: 'flex-start' }}>
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
                {legendUrl && (
                  <div className="heatmap-legend">
                    <img src={legendUrl} alt="Heatmap Legend" />
                  </div>
                )}
              </div>
            </div>

            <div className="analyze-container">
              {tumorFound === false ? (
                <div className="tumor-sizes">
                  <h4>No tumor detected.</h4>
                  <p>This image does not appear to contain a tumor.</p>
                </div>
              ) : (
                <>
                  <div className="tumor-sizes">
                    <h4 style={{ marginBottom: '5px' }}>Detected Tumor Sizes:</h4>
                    {boxes.length === 0 ? (
                      <p>No tumors detected.</p>
                    ) : (
                      getTumorSizes().map(({ id, widthPx, heightPx, widthMM, heightMM }) => (
                        <div key={id}>
                          <h5><strong> Tumor {id}:</strong></h5>
                          <p>
                            {widthPx}px × {heightPx}px
                            <br />
                            ({widthMM}mm × {heightMM}mm)
                          </p>
                        </div>
                      ))
                    )}
                  </div>

                  <div className="toggles-container">
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
                </>
              )}
            </div>
          </div>
        </div>

        <div style={{ marginTop: '30px' }}>
          <button className="top-left-button" onClick={() => navigate('/image')}>
            Back
          </button>
        </div>
        <button className="bottom-right-button" onClick={() => navigate('/upload')}>
          Upload New File
        </button>
      </div>
    </div>
  );
};

export default AnalyzedImage;
