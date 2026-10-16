import React, { useState } from 'react';
import axios from 'axios';

const Login = ({ onLogin }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    first_name: '',
    last_name: '',
    company: ''
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState('');

  // API Base URL - uses React proxy (package.json proxy: http://localhost:5000)
  const API_BASE_URL = '';

  // Handle input changes
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
    
    // Clear errors when user starts typing
    if (error) setError('');
    if (success) setSuccess('');
  };

  // Email validation
  const validateEmail = (email) => {
    const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
    return emailRegex.test(email);
  };

  // Password validation
  const validatePassword = (password) => {
    const minLength = password.length >= 8;
    const hasUpper = /[A-Z]/.test(password);
    const hasLower = /[a-z]/.test(password);
    const hasNumber = /\d/.test(password);
    const hasSpecial = /[!@#$%^&*(),.?":{}|<>]/.test(password);
    
    return minLength && hasUpper && hasLower && hasNumber && hasSpecial;
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccess('');

    try {
      // Client-side validation
      if (!formData.email || !formData.password) {
        setError('Email and password are required');
        setLoading(false);
        return;
      }

      if (!validateEmail(formData.email)) {
        setError('Please enter a valid email address');
        setLoading(false);
        return;
      }

      // Additional validation for registration
      if (!isLogin) {
        if (!validatePassword(formData.password)) {
          setError('Password must be at least 8 characters with uppercase, lowercase, number, and special character');
          setLoading(false);
          return;
        }

        if (!formData.first_name || !formData.last_name) {
          setError('First name and last name are required for registration');
          setLoading(false);
          return;
        }

        if (formData.first_name.length < 2 || formData.last_name.length < 2) {
          setError('Names must be at least 2 characters long');
          setLoading(false);
          return;
        }
      }

      // Prepare API call
      const endpoint = isLogin ? '/api/auth/login' : '/api/auth/register';
      const submitData = isLogin ? 
        { 
          email: formData.email, 
          password: formData.password 
        } : 
        { 
          email: formData.email,
          password: formData.password,
          first_name: formData.first_name,
          last_name: formData.last_name,
          company: formData.company
        };

      console.log('Making API call to:', `${API_BASE_URL}${endpoint}`);
      console.log('With data:', submitData);

      const response = await axios.post(`${API_BASE_URL}${endpoint}`, submitData, {
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: 10000 // 10 second timeout
      });

      console.log('API response:', response.data);

      if (isLogin) {
        // Handle login success
        if (response.data.access_token) {
          // Store token and user data
          localStorage.setItem('token', response.data.access_token);
          localStorage.setItem('user', JSON.stringify(response.data.user));
          
          // Set axios default header for future requests
          axios.defaults.headers.common['Authorization'] = `Bearer ${response.data.access_token}`;
          
          setSuccess('Login successful! Redirecting...');
          
          // Call parent component's onLogin function
          setTimeout(() => {
            onLogin(response.data.user);
          }, 1000);
        }
      } else {
        // Handle registration success
        setSuccess('Registration successful! Please login with your credentials.');
        setIsLogin(true);
        setFormData({
          email: formData.email, // Keep email for convenience
          password: '',
          first_name: '',
          last_name: '',
          company: ''
        });
      }

    } catch (error) {
      console.error('API Error:', error);
      
      // Handle different types of errors
      if (error.code === 'ECONNREFUSED' || error.message.includes('Network Error')) {
        setError('Unable to connect to the authentication service. Please ensure the backend is running on port 5000.');
      } else if (error.response) {
        // Server responded with error status
        const errorData = error.response.data;
        if (errorData.details && Array.isArray(errorData.details)) {
          setError(errorData.details.join(', '));
        } else if (errorData.error) {
          setError(errorData.error);
        } else {
          setError('Authentication failed. Please try again.');
        }
      } else if (error.request) {
        // Request was made but no response received
        setError('No response from server. Please check your internet connection.');
      } else {
        // Something else happened
        setError('An unexpected error occurred. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  // Toggle between login and register
  const toggleMode = () => {
    setIsLogin(!isLogin);
    setError('');
    setSuccess('');
    setFormData({
      email: '',
      password: '',
      first_name: '',
      last_name: '',
      company: ''
    });
  };

  return (
    <div className="container mt-5">
      <div className="row justify-content-center">
        <div className="col-md-6 col-lg-4">
          <div className="card shadow">
            <div className="card-header bg-primary text-white">
              <h3 className="text-center mb-0">
                <i className="fas fa-shield-alt me-2"></i>
                {isLogin ? 'Login' : 'Register'}
              </h3>
            </div>
            <div className="card-body">
              {/* Success Message */}
              {success && (
                <div className="alert alert-success" role="alert">
                  <i className="fas fa-check-circle me-2"></i>
                  {success}
                </div>
              )}

              {/* Error Message */}
              {error && (
                <div className="alert alert-danger" role="alert">
                  <i className="fas fa-exclamation-triangle me-2"></i>
                  {error}
                </div>
              )}

              <form onSubmit={handleSubmit}>
                {/* Email Field */}
                <div className="mb-3">
                  <label htmlFor="email" className="form-label">
                    <i className="fas fa-envelope me-1"></i>
                    Email *
                  </label>
                  <input
                    type="email"
                    className="form-control"
                    id="email"
                    name="email"
                    value={formData.email}
                    onChange={handleInputChange}
                    required
                    placeholder="Enter your email"
                    disabled={loading}
                  />
                </div>

                {/* Password Field */}
                <div className="mb-3">
                  <label htmlFor="password" className="form-label">
                    <i className="fas fa-lock me-1"></i>
                    Password *
                  </label>
                  <input
                    type="password"
                    className="form-control"
                    id="password"
                    name="password"
                    value={formData.password}
                    onChange={handleInputChange}
                    required
                    placeholder="Enter your password"
                    disabled={loading}
                  />
                  {!isLogin && (
                    <div className="form-text">
                      Must be 8+ characters with uppercase, lowercase, number, and special character
                    </div>
                  )}
                </div>

                {/* Registration Fields */}
                {!isLogin && (
                  <>
                    <div className="row">
                      <div className="col-md-6 mb-3">
                        <label htmlFor="first_name" className="form-label">
                          <i className="fas fa-user me-1"></i>
                          First Name *
                        </label>
                        <input
                          type="text"
                          className="form-control"
                          id="first_name"
                          name="first_name"
                          value={formData.first_name}
                          onChange={handleInputChange}
                          required
                          placeholder="First name"
                          disabled={loading}
                        />
                      </div>
                      <div className="col-md-6 mb-3">
                        <label htmlFor="last_name" className="form-label">
                          <i className="fas fa-user me-1"></i>
                          Last Name *
                        </label>
                        <input
                          type="text"
                          className="form-control"
                          id="last_name"
                          name="last_name"
                          value={formData.last_name}
                          onChange={handleInputChange}
                          required
                          placeholder="Last name"
                          disabled={loading}
                        />
                      </div>
                    </div>

                    <div className="mb-3">
                      <label htmlFor="company" className="form-label">
                        <i className="fas fa-building me-1"></i>
                        Company
                      </label>
                      <input
                        type="text"
                        className="form-control"
                        id="company"
                        name="company"
                        value={formData.company}
                        onChange={handleInputChange}
                        placeholder="Company name (optional)"
                        disabled={loading}
                      />
                    </div>
                  </>
                )}

                {/* Submit Button */}
                <button
                  type="submit"
                  className={`btn btn-primary w-100 ${loading ? 'disabled' : ''}`}
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <i className="fas fa-spinner fa-spin me-2"></i>
                      {isLogin ? 'Signing In...' : 'Creating Account...'}
                    </>
                  ) : (
                    <>
                      <i className={`fas ${isLogin ? 'fa-sign-in-alt' : 'fa-user-plus'} me-2`}></i>
                      {isLogin ? 'Sign In' : 'Create Account'}
                    </>
                  )}
                </button>
              </form>

              {/* Toggle Mode */}
              <div className="text-center mt-3">
                <button
                  type="button"
                  className="btn btn-link text-decoration-none"
                  onClick={toggleMode}
                  disabled={loading}
                >
                  {isLogin 
                    ? "Don't have an account? Sign up" 
                    : "Already have an account? Sign in"
                  }
                </button>
              </div>


            </div>
          </div>
        </div>
      </div>

      {/* Connection Status */}
      <div className="row justify-content-center mt-3">
        <div className="col-md-6 col-lg-4">
          <div className="text-center">
            <small className="text-muted">
              <i className="fas fa-server me-1"></i>
              Authentication Service: {API_BASE_URL || 'http://localhost:5000'}
            </small>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;