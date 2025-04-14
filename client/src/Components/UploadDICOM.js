import React, { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';

const UploadDICOM = ({ setDicomData, setMaskImage, setTumorFound }) => {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('No file chosen');
  const [error, setError] = useState('');
  const navigate = useNavigate();
  const fileInputRef = useRef();
  const [loading, setLoading] = useState(false);

  const onDrop = (acceptedFiles) => {
    const uploadedFile = acceptedFiles[0];
    if (uploadedFile && uploadedFile.name.endsWith('.dcm')) {
      setFile(uploadedFile);
      setFileName(uploadedFile.name);
      setError('');
    } else {
      setFile(null);
      setFileName('No file chosen');
      setError('Please upload a valid DICOM (.dcm) file.');
    }
  };

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: { 'application/dicom': ['.dcm'] },
    multiple: false,
  });

  const handleUpload = async () => {
    setLoading(true);
    if (!file) {
      setError('Please select a file to upload.');
      setLoading(false);
      return;
    }

    const formData = new FormData();
    formData.append('dicomFile', file);

    try {
      const response = await fetch('http://localhost:5000/upload-dicom', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const result = await response.json();
        setError(result.error || 'Unknown error occurred.');
        return;
      }

      const data = await response.json(); 

      // Reset previous analysis data in localStorage
      localStorage.removeItem('maskImage');
      localStorage.removeItem('tumorFound');

      setMaskImage(null);
      setTumorFound(null);

      // Update state with new file
      setDicomData({
        imageUrl: data.image,          // base64 image string
        fileName: fileName,
        patientInfo: data.patient      // contains id, sex, age
      });

      navigate('/image');
    } catch (err) {
      setError('Error uploading file: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-dicom">
      <h2 className="upload-title">Upload DICOM File</h2>

      <div {...getRootProps()} className="dropzone">
        <input {...getInputProps()} />
        <p>Drag & drop your DICOM file here, or click to select a file</p>
      </div>

      <input
        type="file"
        accept=".dcm"
        ref={fileInputRef}
        style={{ display: 'none' }}
        onChange={(e) => {
          const selectedFile = e.target.files[0];
          if (selectedFile) {
            setFile(selectedFile);
            setFileName(selectedFile.name);
            setError('');
          } else {
            setFile(null);
            setFileName('No file chosen');
          }
        }}
      />

      <button
        className="choose-file-button"
        onClick={() => fileInputRef.current.click()}
        type="button"
      >
        Choose File
      </button>

      <p className="file-name">{fileName}</p>
      {error && <p className="error-text">{error}</p>}

      <div>
        {loading ? (
          <div className="loading-container">
            <span className="loader" style={{ paddingTop: "5px" }}></span>
            <p style={{ paddingTop: "5px" }}>Uploading</p>
          </div>
        ) : (
          <button
            className={`upload-button ${!file || loading ? 'disabled' : ''}`}
            onClick={handleUpload}
            disabled={!file || loading}
          >
            Upload DICOM
          </button>
        )}
      </div>
    </div>
  );
};

export default UploadDICOM;
