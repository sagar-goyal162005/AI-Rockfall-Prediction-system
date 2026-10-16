import React, { useState, useEffect } from 'react';
import axios from 'axios';

const Reports = () => {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  useEffect(() => {
    fetchReports();
  }, []);

  const fetchReports = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/reports/history');
      setReports(response.data.reports);
    } catch (error) {
      setError('Failed to fetch reports');
      console.error('Error fetching reports:', error);
    } finally {
      setLoading(false);
    }
  };

  const generateReport = async (reportType) => {
    try {
      setGenerating(true);
      setError('');
      setSuccess('');

      const response = await axios.post('/api/reports/generate', {
        report_type: reportType
      });

      setSuccess(`${reportType.charAt(0).toUpperCase() + reportType.slice(1)} report generation requested. You will receive it via email shortly.`);
      
      // Refresh reports list
      setTimeout(() => {
        fetchReports();
      }, 2000);

    } catch (error) {
      setError(error.response?.data?.error || 'Failed to generate report');
    } finally {
      setGenerating(false);
    }
  };

  const getStatusBadge = (status) => {
    const statusConfig = {
      'generating': { class: 'warning', text: 'Generating' },
      'sent': { class: 'success', text: 'Sent' },
      'failed': { class: 'danger', text: 'Failed' }
    };

    const config = statusConfig[status] || { class: 'secondary', text: status };
    return <span className={`badge bg-${config.class}`}>{config.text}</span>;
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString();
  };

  return (
    <div className="reports-container">
      <div className="reports-header">
        <h2>Risk Analysis Reports</h2>
        <p>Generate and manage your rockfall risk analysis reports</p>
      </div>

      {error && (
        <div className="alert alert-danger">
          {error}
        </div>
      )}

      {success && (
        <div className="alert alert-success">
          {success}
        </div>
      )}

      {/* Report Generation Section */}
      <div className="card report-generation-card">
        <div className="card-header">
          <h4>Generate New Report</h4>
        </div>
        <div className="card-body">
          <div className="row">
            <div className="col-md-4">
              <div className="report-type-card">
                <h5>Daily Report</h5>
                <p>Comprehensive analysis of today's risk assessments and sensor data</p>
                <button
                  className="btn btn-primary"
                  onClick={() => generateReport('daily')}
                  disabled={generating}
                >
                  {generating ? (
                    <>
                      <span className="spinner-border spinner-border-sm me-2"></span>
                      Generating...
                    </>
                  ) : (
                    'Generate Daily Report'
                  )}
                </button>
              </div>
            </div>
            <div className="col-md-4">
              <div className="report-type-card">
                <h5>Weekly Report</h5>
                <p>Weekly summary with trends and pattern analysis</p>
                <button
                  className="btn btn-primary"
                  onClick={() => generateReport('weekly')}
                  disabled={generating}
                >
                  {generating ? (
                    <>
                      <span className="spinner-border spinner-border-sm me-2"></span>
                      Generating...
                    </>
                  ) : (
                    'Generate Weekly Report'
                  )}
                </button>
              </div>
            </div>
            <div className="col-md-4">
              <div className="report-type-card">
                <h5>Monthly Report</h5>
                <p>Complete monthly overview with detailed recommendations</p>
                <button
                  className="btn btn-primary"
                  onClick={() => generateReport('monthly')}
                  disabled={generating}
                >
                  {generating ? (
                    <>
                      <span className="spinner-border spinner-border-sm me-2"></span>
                      Generating...
                    </>
                  ) : (
                    'Generate Monthly Report'
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Report History Section */}
      <div className="card report-history-card">
        <div className="card-header d-flex justify-content-between align-items-center">
          <h4>Report History</h4>
          <button className="btn btn-outline-primary btn-sm" onClick={fetchReports}>
            <i className="fas fa-refresh me-2"></i>
            Refresh
          </button>
        </div>
        <div className="card-body">
          {loading ? (
            <div className="text-center py-4">
              <div className="spinner-border text-primary" role="status">
                <span className="visually-hidden">Loading...</span>
              </div>
              <p className="mt-2">Loading reports...</p>
            </div>
          ) : reports.length === 0 ? (
            <div className="text-center py-4">
              <i className="fas fa-file-alt fa-3x text-muted mb-3"></i>
              <p className="text-muted">No reports generated yet. Generate your first report above!</p>
            </div>
          ) : (
            <div className="table-responsive">
              <table className="table table-hover">
                <thead>
                  <tr>
                    <th>Report Type</th>
                    <th>Status</th>
                    <th>Requested</th>
                    <th>Generated</th>
                    <th>Email Sent</th>
                  </tr>
                </thead>
                <tbody>
                  {reports.map((report) => (
                    <tr key={report.id}>
                      <td>
                        <span className="fw-bold">
                          {report.report_type.charAt(0).toUpperCase() + report.report_type.slice(1)}
                        </span>
                      </td>
                      <td>{getStatusBadge(report.status)}</td>
                      <td>{formatDate(report.requested_at)}</td>
                      <td>
                        {report.generated_at ? formatDate(report.generated_at) : '-'}
                      </td>
                      <td>
                        {report.email_sent ? (
                          <span className="text-success">
                            <i className="fas fa-check-circle me-1"></i>
                            Yes
                          </span>
                        ) : (
                          <span className="text-muted">
                            <i className="fas fa-times-circle me-1"></i>
                            No
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      <style jsx>{`
        .reports-container {
          padding: 2rem;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          min-height: 100vh;
        }

        .reports-header {
          text-align: center;
          margin-bottom: 2rem;
          color: #fff;
        }

        .reports-header h2 {
          font-size: 2.5rem;
          font-weight: 700;
          margin-bottom: 0.5rem;
          text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }

        .reports-header p {
          font-size: 1.1rem;
          opacity: 0.9;
          text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }

        .card {
          background: rgba(255, 255, 255, 0.1);
          backdrop-filter: blur(20px);
          border: 1px solid rgba(255, 255, 255, 0.2);
          border-radius: 20px;
          box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
          margin-bottom: 2rem;
          overflow: hidden;
        }

        .card-header {
          background: rgba(255, 255, 255, 0.1);
          border-bottom: 1px solid rgba(255, 255, 255, 0.2);
          padding: 1.5rem;
        }

        .card-header h4 {
          color: #fff;
          margin: 0;
          font-weight: 600;
          text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }

        .card-body {
          padding: 2rem;
        }

        .report-type-card {
          background: rgba(255, 255, 255, 0.1);
          border: 1px solid rgba(255, 255, 255, 0.2);
          border-radius: 15px;
          padding: 1.5rem;
          text-align: center;
          height: 100%;
          transition: all 0.3s ease;
        }

        .report-type-card:hover {
          background: rgba(255, 255, 255, 0.15);
          transform: translateY(-5px);
          box-shadow: 0 15px 30px rgba(0, 0, 0, 0.2);
        }

        .report-type-card h5 {
          color: #fff;
          font-weight: 600;
          margin-bottom: 1rem;
          text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }

        .report-type-card p {
          color: rgba(255, 255, 255, 0.9);
          margin-bottom: 1.5rem;
          text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
        }

        .btn-primary {
          background: linear-gradient(135deg, #4a90e2 0%, #50c878 100%);
          border: none;
          border-radius: 10px;
          padding: 0.75rem 1.5rem;
          font-weight: 600;
          transition: all 0.3s ease;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .btn-primary:hover {
          transform: translateY(-2px);
          box-shadow: 0 10px 20px rgba(74, 144, 226, 0.3);
        }

        .btn-outline-primary {
          border: 2px solid rgba(255, 255, 255, 0.7);
          color: #fff;
          background: transparent;
          border-radius: 10px;
          transition: all 0.3s ease;
        }

        .btn-outline-primary:hover {
          background: rgba(255, 255, 255, 0.1);
          border-color: #fff;
          color: #fff;
        }

        .table {
          color: #fff;
          background: transparent;
        }

        .table th {
          border-color: rgba(255, 255, 255, 0.2);
          background: rgba(255, 255, 255, 0.1);
          color: #fff;
          font-weight: 600;
          text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
        }

        .table td {
          border-color: rgba(255, 255, 255, 0.1);
          color: rgba(255, 255, 255, 0.9);
        }

        .table-hover tbody tr:hover {
          background: rgba(255, 255, 255, 0.1);
        }

        .badge {
          font-size: 0.75rem;
          padding: 0.5rem 0.75rem;
          border-radius: 10px;
        }

        .alert {
          border: none;
          border-radius: 15px;
          backdrop-filter: blur(10px);
          margin-bottom: 1.5rem;
        }

        .alert-danger {
          background: rgba(255, 107, 107, 0.2);
          color: #ff6b6b;
          border: 1px solid rgba(255, 107, 107, 0.3);
        }

        .alert-success {
          background: rgba(80, 200, 120, 0.2);
          color: #50c878;
          border: 1px solid rgba(80, 200, 120, 0.3);
        }

        .text-success {
          color: #50c878 !important;
        }

        .text-muted {
          color: rgba(255, 255, 255, 0.6) !important;
        }

        .spinner-border-sm {
          width: 1rem;
          height: 1rem;
        }
      `}</style>
    </div>
  );
};

export default Reports;