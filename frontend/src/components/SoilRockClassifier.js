import React, { useState, useRef } from 'react';
import './SoilRockClassifier.css';

const SoilRockClassifier = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [classification, setClassification] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleImageSelect = (file) => {
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setSelectedImage({
          file: file,
          url: e.target.result,
          name: file.name
        });
        setClassification(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleFileInputChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      handleImageSelect(file);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleImageSelect(e.dataTransfer.files[0]);
    }
  };

  const classifyImage = async () => {
    if (!selectedImage) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('image', selectedImage.file);

    try {
      const response = await fetch('/api/soil-rock/classify', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        setClassification(result);
      } else {
        console.error('Classification failed');
        setClassification({
          error: 'Classification failed. Please try again.',
          predicted_class: 'Unknown',
          confidence: 0
        });
      }
    } catch (error) {
      console.error('Error:', error);
      setClassification({
        error: 'Network error. Please check your connection.',
        predicted_class: 'Unknown',
        confidence: 0
      });
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case 'low': return '#28a745';
      case 'medium': return '#ffc107';
      case 'high': return '#fd7e14';
      case 'critical': return '#dc3545';
      default: return '#6c757d';
    }
  };

  return (
    <div className="soil-rock-classifier">
      <div className="classifier-header">
        <h2>🪨 Soil & Rock Classification System</h2>
        <p>Upload an image to identify soil/rock type and get detailed geological analysis</p>
      </div>

      <div className="classifier-content">
        {/* Upload Section */}
        <div className="upload-section">
          <div
            className={`upload-area ${dragActive ? 'drag-active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileInputChange}
              style={{ display: 'none' }}
            />
            
            {selectedImage ? (
              <div className="image-preview">
                <img src={selectedImage.url} alt="Selected" />
                <div className="image-info">
                  <p>{selectedImage.name}</p>
                  <button className="change-image-btn" onClick={(e) => {
                    e.stopPropagation();
                    fileInputRef.current?.click();
                  }}>
                    Change Image
                  </button>
                </div>
              </div>
            ) : (
              <div className="upload-placeholder">
                <div className="upload-icon">📷</div>
                <h3>Upload Soil/Rock Image</h3>
                <p>Drag and drop an image here, or click to select</p>
                <p className="supported-formats">Supported: JPG, PNG, JPEG</p>
              </div>
            )}
          </div>

          {selectedImage && (
            <div className="action-buttons">
              <button
                className="classify-btn"
                onClick={classifyImage}
                disabled={loading}
              >
                {loading ? (
                  <>
                    <div className="loading-spinner"></div>
                    Analyzing...
                  </>
                ) : (
                  <>
                    🔍 Classify Soil/Rock
                  </>
                )}
              </button>
            </div>
          )}
        </div>

        {/* Results Section */}
        {classification && (
          <div className="results-section">
            <div className="classification-result">
              <div className="result-header">
                <h3>🎯 Classification Results</h3>
                <div className="confidence-badge">
                  Confidence: {classification.confidence}%
                </div>
              </div>

              <div className="predicted-class">
                <h4>Identified Type:</h4>
                <div className="class-name">{classification.predicted_class}</div>
              </div>

              {/* Wikipedia Information */}
              {classification.wikipedia_info && (
                <div className="wikipedia-section">
                  <h4>📚 Wikipedia Information</h4>
                  <div className="wiki-content">
                    <p className="wiki-description">{classification.wikipedia_info.description}</p>
                    {classification.wikipedia_info.properties && (
                      <div className="properties">
                        <h5>Properties:</h5>
                        <ul>
                          {classification.wikipedia_info.properties.map((prop, index) => (
                            <li key={index}>{prop}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {classification.wikipedia_info.uses && (
                      <div className="uses">
                        <h5>Common Uses:</h5>
                        <ul>
                          {classification.wikipedia_info.uses.map((use, index) => (
                            <li key={index}>{use}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {classification.wikipedia_info.url && (
                      <a 
                        href={classification.wikipedia_info.url} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="wiki-link"
                      >
                        📖 Read more on Wikipedia
                      </a>
                    )}
                  </div>
                </div>
              )}

              {/* Geological Details */}
              {classification.geological_details && (
                <div className="geological-section">
                  <h4>🏔️ Geological Analysis</h4>
                  <div className="geo-grid">
                    <div className="geo-item">
                      <span className="geo-label">Formation Type:</span>
                      <span className="geo-value">{classification.geological_details.formation_type}</span>
                    </div>
                    <div className="geo-item">
                      <span className="geo-label">Hardness:</span>
                      <span className="geo-value">{classification.geological_details.hardness}</span>
                    </div>
                    <div className="geo-item">
                      <span className="geo-label">Porosity:</span>
                      <span className="geo-value">{classification.geological_details.porosity}</span>
                    </div>
                    <div className="geo-item">
                      <span className="geo-label">Mining Suitability:</span>
                      <span className="geo-value">{classification.geological_details.mining_suitability}</span>
                    </div>
                  </div>
                </div>
              )}

              {classification.analysis && (
                <div className="detailed-info">
                  <div className="info-grid">
                    {/* Risk Assessment */}
                    <div className="info-card risk-assessment">
                      <h5>⚠️ Mining Risk Assessment</h5>
                      <div className="risk-level">
                        <span 
                          className="risk-indicator"
                          style={{ color: getRiskColor(classification.analysis.risk_level) }}
                        >
                          {classification.analysis.risk_level}
                        </span>
                      </div>
                      <p className="stability-info">
                        <strong>Stability:</strong> {classification.analysis.stability_assessment}
                      </p>
                    </div>

                    {/* Recommendations */}
                    <div className="info-card recommendations">
                      <h5>💡 Recommendations</h5>
                      <p>{classification.analysis.recommendations}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SoilRockClassifier;