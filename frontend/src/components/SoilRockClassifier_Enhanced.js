import React, { useState, useRef } from 'react';
import './SoilRockClassifier.css';

const SoilRockClassifier = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [classification, setClassification] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState('');
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
        setError('');
      };
      reader.readAsDataURL(file);
    } else {
      setError('Please select a valid image file (JPG, PNG, GIF, etc.)');
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
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError('');
    
    const formData = new FormData();
    formData.append('image', selectedImage.file);

    try {
      console.log('Sending classification request...');
      const response = await fetch('/api/soil-rock/classify', {
        method: 'POST',
        body: formData,
      });

      console.log('Response status:', response.status);

      if (response.ok) {
        const result = await response.json();
        console.log('Classification result:', result);
        
        // Check if the result contains an error field (from server error responses)
        if (result.error) {
          setError(result.error);
          setClassification(null);
        } else if (result.predicted_class) {
          // Valid classification response - has the required fields
          // Add missing fields for UI compatibility
          const enhancedResult = {
            ...result,
            confidence_level: result.confidence >= 80 ? 'High' : 
                             result.confidence >= 60 ? 'Medium' : 'Low'
          };
          setClassification(enhancedResult);
          setError('');
        } else {
          setError('Invalid response format from server');
          setClassification(null);
        }
      } else {
        const errorText = await response.text();
        console.error('Classification failed:', errorText);
        setError(`Classification failed: ${response.status} ${response.statusText}`);
        setClassification(null);
      }
    } catch (error) {
      console.error('Network error:', error);
      setError(`Network error: ${error.message}. Please ensure the backend server is running on port 5000.`);
      setClassification(null);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (riskLevel) => {
    switch (riskLevel?.toUpperCase()) {
      case 'LOW': return '#28a745';
      case 'MEDIUM': return '#ffc107';
      case 'HIGH': return '#fd7e14';
      case 'CRITICAL': return '#dc3545';
      default: return '#6c757d';
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 90) return '#28a745';
    if (confidence >= 75) return '#17a2b8';
    if (confidence >= 60) return '#ffc107';
    return '#dc3545';
  };

  return (
    <div className="soil-rock-classifier">
      <div className="classifier-header">
        <h2>🪨 AI-Powered Soil & Rock Classification System</h2>
        <p>Upload an image to identify rock/soil type with comprehensive geological analysis and Wikipedia integration</p>
      </div>

      <div className="classifier-content">
        {/* Upload Section */}
        <div className="upload-section">
          <div
            className={`upload-area ${dragActive ? 'drag-active' : ''} ${selectedImage ? 'has-image' : ''}`}
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
                <img src={selectedImage.url} alt="Selected" className="preview-image" />
                <div className="image-info">
                  <p><strong>File:</strong> {selectedImage.name}</p>
                  <button 
                    className="btn btn-primary classify-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      classifyImage();
                    }}
                    disabled={loading}
                  >
                    {loading ? (
                      <>
                        <span className="spinner-border spinner-border-sm me-2"></span>
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <i className="fas fa-microscope me-2"></i>
                        Classify Rock/Soil
                      </>
                    )}
                  </button>
                </div>
              </div>
            ) : (
              <div className="upload-placeholder">
                <i className="fas fa-cloud-upload-alt"></i>
                <h4>Upload Rock/Soil Image</h4>
                <p>Drag & drop an image here or click to browse</p>
                <small>Supports: JPG, PNG, GIF (Max 10MB)</small>
              </div>
            )}
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="alert alert-danger mt-3">
            <i className="fas fa-exclamation-triangle me-2"></i>
            {error}
          </div>
        )}

        {/* Classification Results */}
        {classification && (
          <div className="classification-results">
            <div className="results-header">
              <h3>
                <i className="fas fa-chart-line me-2"></i>
                Classification Results
              </h3>
              <div 
                className="confidence-badge"
                style={{ backgroundColor: getConfidenceColor(classification.confidence) }}
              >
                Confidence: {classification.confidence}%
              </div>
            </div>

            {/* Main Result */}
            <div className="main-result">
              <div className="result-card">
                <h4>Identified Type:</h4>
                <div className="rock-type-result">
                  <span className="rock-name">{classification.predicted_class}</span>
                  <span className={`confidence-level ${classification.confidence_level?.toLowerCase()}`}>
                    {classification.confidence_level} Confidence
                  </span>
                </div>
              </div>
            </div>

            {/* Wikipedia Information */}
            {classification.wikipedia_info && (
              <div className="wikipedia-section">
                <h4><i className="fab fa-wikipedia-w me-2"></i>Wikipedia Information</h4>
                <div className="wikipedia-content">
                  <div className="wiki-text">
                    <h5>{classification.wikipedia_info.title || classification.predicted_class}</h5>
                    <p>{classification.wikipedia_info.description}</p>
                    {classification.wikipedia_info.url && (
                      <a 
                        href={classification.wikipedia_info.url} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="btn btn-outline-primary btn-sm"
                      >
                        <i className="fas fa-external-link-alt me-1"></i>
                        Read More on Wikipedia
                      </a>
                    )}
                  </div>
                  {classification.wikipedia_info.thumbnail && (
                    <div className="wiki-thumbnail">
                      <img 
                        src={classification.wikipedia_info.thumbnail} 
                        alt={classification.wikipedia_info.title}
                        className="thumbnail-img"
                      />
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Detailed Analysis */}
            {(classification.geological_details || classification.analysis) && (
              <div className="detailed-analysis">
                <h4><i className="fas fa-microscope me-2"></i>Comprehensive Geological Analysis</h4>
                
                {/* Physical Properties */}
                {classification.geological_details && (
                  <div className="analysis-grid">
                    <div className="analysis-card">
                      <h5>🔬 Physical Properties</h5>
                      <div className="property-list">
                        <div className="property-item">
                          <span className="label">Formation Type:</span>
                          <span className="value">{classification.geological_details.formation_type}</span>
                        </div>
                        <div className="property-item">
                          <span className="label">Hardness Scale:</span>
                          <span className="value">{classification.geological_details.hardness}</span>
                        </div>
                        <div className="property-item">
                          <span className="label">Porosity:</span>
                          <span className="value">{classification.geological_details.porosity}</span>
                        </div>
                        <div className="property-item">
                          <span className="label">Density:</span>
                          <span className="value">{classification.geological_details.density}</span>
                        </div>
                        <div className="property-item">
                          <span className="label">Compressive Strength:</span>
                          <span className="value">{classification.geological_details.compressive_strength}</span>
                        </div>
                        <div className="property-item">
                          <span className="label">Mining Suitability:</span>
                          <span className="value">{classification.geological_details.mining_suitability}</span>
                        </div>
                      </div>
                    </div>

                    {/* Risk Assessment */}
                    {classification.analysis && (
                      <div className="analysis-card">
                        <h5>⚠️ Risk Assessment & Engineering Analysis</h5>
                        <div className="risk-analysis">
                          <div className="risk-level">
                            <span className="label">Risk Level:</span>
                            <span 
                              className="risk-badge"
                              style={{ backgroundColor: getRiskColor(classification.analysis.risk_level) }}
                            >
                              {classification.analysis.risk_level}
                            </span>
                          </div>
                          <div className="property-item">
                            <span className="label">Stability Assessment:</span>
                            <span className="value">{classification.analysis.stability_assessment}</span>
                          </div>
                          <div className="property-item">
                            <span className="label">Engineering Notes:</span>
                            <span className="value">{classification.analysis.engineering_notes}</span>
                          </div>
                          <div className="property-item">
                            <span className="label">Recommendations:</span>
                            <span className="value">{classification.analysis.recommendations}</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Uses and Applications */}
                {classification.wikipedia_info && classification.wikipedia_info.uses && (
                  <div className="uses-section">
                    <h5>🏗️ Uses & Applications</h5>
                    <div className="uses-grid">
                      {classification.wikipedia_info.uses.map((use, index) => (
                        <div key={index} className="use-item">
                          <i className="fas fa-check-circle me-2"></i>
                          {use}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Additional Properties */}
                {classification.wikipedia_info && classification.wikipedia_info.properties && (
                  <div className="properties-section">
                    <h5>📋 Additional Properties</h5>
                    <div className="properties-grid">
                      {classification.wikipedia_info.properties.map((property, index) => (
                        <div key={index} className="property-badge">
                          {property}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default SoilRockClassifier;