import os
import sys
import io

if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from datetime import datetime, timedelta
import json
import numpy as np
import random
from dotenv import load_dotenv
import time
from werkzeug.utils import secure_filename
import pandas as pd
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import uuid
 
# Add the path for our LSTM model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lidar_processing'))

# Import auto-training functionality
sys.path.append(os.path.dirname(__file__))
try:
    from auto_model_trainer import AutoModelTrainer
    auto_trainer = AutoModelTrainer()
    AUTO_TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"Auto training not available: {e}")
    AUTO_TRAINING_AVAILABLE = False
    auto_trainer = None

# Import new services
try:
    from soil_rock_classifier import soil_rock_classifier
    SOIL_CLASSIFICATION_AVAILABLE = True
    print("[OK] Soil/Rock classification service loaded")
except ImportError as e:
    print(f"Soil/Rock classification not available: {e}")
    SOIL_CLASSIFICATION_AVAILABLE = False

try:
    from email_report_service import email_report_service
    EMAIL_REPORTING_AVAILABLE = True
    print("[OK] Email reporting service loaded")
except ImportError as e:
    print(f"Email reporting not available: {e}")
    EMAIL_REPORTING_AVAILABLE = False

# Look for deep_learning_model in multiple locations
import sys
import os

# Add potential paths for the deep learning model
possible_paths = [
    os.path.join(os.path.dirname(__file__), '..', 'lidar_processing'),
    os.path.join(os.path.dirname(__file__), '..', 'ml_models'),
    os.path.dirname(__file__)
]

for path in possible_paths:
    if path not in sys.path:
        sys.path.append(path)

try:
    from deep_learning_model import RockfallLSTMPredictor, LSTMConfig
    LSTM_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: Deep learning model not available. LSTM features disabled.")
    LSTM_AVAILABLE = False

load_dotenv()

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Authentication configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'jwt-secret-change-in-production')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///rockfall_system.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy()
bcrypt = Bcrypt()
jwt = JWTManager()

db.init_app(app)
bcrypt.init_app(app)
jwt.init_app(app)

# Database Models (inline for simplicity)
class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    company = db.Column(db.String(100))
    role = db.Column(db.String(50), default='user')
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    
    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'company': self.company,
            'role': self.role,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }

class ReportRequest(db.Model):
    __tablename__ = 'report_requests'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    report_type = db.Column(db.String(50), nullable=False)  # 'daily', 'weekly', 'monthly'
    status = db.Column(db.String(20), default='pending')  # 'pending', 'generating', 'sent', 'failed'
    email_sent = db.Column(db.Boolean, default=False)
    requested_at = db.Column(db.DateTime, default=datetime.utcnow)
    generated_at = db.Column(db.DateTime)
    file_path = db.Column(db.String(255))
    
    user = db.relationship('User', backref=db.backref('report_requests', lazy=True))

class SensorData(db.Model):
    __tablename__ = 'sensor_data'
    
    id = db.Column(db.Integer, primary_key=True)
    sensor_id = db.Column(db.String(50), nullable=False)
    sensor_type = db.Column(db.String(50), nullable=False)
    location_x = db.Column(db.Float, nullable=False)
    location_y = db.Column(db.Float, nullable=False)
    value = db.Column(db.Float, nullable=False)
    unit = db.Column(db.String(20), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class RiskAssessment(db.Model):
    __tablename__ = 'risk_assessments'
    
    id = db.Column(db.Integer, primary_key=True)
    risk_level = db.Column(db.String(20), nullable=False)
    probability = db.Column(db.Float, nullable=False)
    affected_zones = db.Column(db.Text, nullable=False)
    recommendations = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class Alert(db.Model):
    __tablename__ = 'alerts'
    
    id = db.Column(db.Integer, primary_key=True)
    alert_type = db.Column(db.String(50), nullable=False)
    severity = db.Column(db.String(20), nullable=False)
    message = db.Column(db.Text, nullable=False)
    location = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='ACTIVE')

class LIDARScan(db.Model):
    __tablename__ = 'lidar_scans'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_format = db.Column(db.String(20), nullable=False)
    num_points = db.Column(db.Integer, nullable=False)
    scan_date = db.Column(db.DateTime)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    bounds_json = db.Column(db.Text)  # JSON string of bounding box
    coordinate_system = db.Column(db.String(100))
    scanner_info = db.Column(db.Text)  # JSON string
    processing_status = db.Column(db.String(50), default='UPLOADED')
    
class LIDARAnalysis(db.Model):
    __tablename__ = 'lidar_analyses'
    
    id = db.Column(db.Integer, primary_key=True)
    scan_id = db.Column(db.Integer, db.ForeignKey('lidar_scans.id'), nullable=False)
    analysis_type = db.Column(db.String(50), nullable=False)  # 'geological', 'risk_assessment', 'feature_detection'
    results_json = db.Column(db.Text, nullable=False)  # JSON string of analysis results
    stability_score = db.Column(db.Float)
    risk_level = db.Column(db.String(20))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    processing_time = db.Column(db.Float)  # seconds
    
    # Relationship
    scan = db.relationship('LIDARScan', backref=db.backref('analyses', lazy=True))

# Simple ML Predictor (inline)
class SimpleRockfallPredictor: 
    def predict_rockfall_risk(self, sensor_data=None):
        # Generate mock prediction
        probability = random.uniform(0.2, 0.8)
        
        if probability < 0.3:
            risk_level = 'LOW'
        elif probability < 0.5:
            risk_level = 'MEDIUM'
        elif probability < 0.7:
            risk_level = 'HIGH'
        else:
            risk_level = 'CRITICAL'
        
        affected_zones = [
            {'lat': -23.5505, 'lng': -46.6333, 'radius': 50, 'risk_level': int(probability * 10)},
            {'lat': -23.5515, 'lng': -46.6343, 'radius': 75, 'risk_level': int(probability * 8)}
        ]
        
        recommendations = [
            f"Current risk level: {risk_level}",
            "Monitor conditions closely",
            "Follow safety protocols"
        ]
        
        if risk_level in ['HIGH', 'CRITICAL']:
            recommendations.extend([
                "Consider restricting access to high-risk areas",
                "Increase monitoring frequency"
            ])
        
        return {
            'risk_level': risk_level,
            'probability': probability,
            'affected_zones': affected_zones,
            'recommendations': recommendations,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def generate_risk_map(self, sensor_data_list):
        risk_zones = []
        for i in range(8):
            lat = -23.5505 + (i * 0.001)
            lng = -46.6333 + (i * 0.001)
            risk_value = random.uniform(0, 1)
            
            risk_zones.append({
                'lat': lat,
                'lng': lng,
                'risk_value': risk_value,
                'risk_level': 'HIGH' if risk_value > 0.7 else 'MEDIUM' if risk_value > 0.4 else 'LOW'
            })
        
        return risk_zones
    
    def generate_forecast(self, days=7):
        forecast_data = {
            'dates': [],
            'probabilities': [],
            'confidence_intervals': []
        }
        
        base_date = datetime.utcnow()
        base_prob = random.uniform(0.3, 0.6)
        
        for i in range(days):
            date = base_date + timedelta(days=i)
            prob = base_prob + (i * 0.02) + random.uniform(-0.05, 0.05)
            prob = max(0, min(1, prob))
            
            forecast_data['dates'].append(date.isoformat())
            forecast_data['probabilities'].append(prob)
            forecast_data['confidence_intervals'].append([
                max(0, prob - 0.1),
                min(1, prob + 0.1)
            ])
        
        return forecast_data

# Initialize services
predictor = SimpleRockfallPredictor()
print("[OK] Services initialized successfully")

# Initialize LIDAR processing services
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from lidar_processing import (
        LIDARFileHandler, 
        PointCloudProcessor, 
        GeologicalFeatureExtractor,
        PointCloudDLModel
    )
    
    lidar_handler = LIDARFileHandler()
    point_cloud_processor = PointCloudProcessor()
    geological_extractor = GeologicalFeatureExtractor()
    dl_model = PointCloudDLModel()
    
    # Skip training for faster startup - train manually if needed
    # dl_model.train_risk_classifier(num_epochs=5)
    
    print("[OK] LIDAR processing services initialized")
except Exception as e:
    print(f"⚠️ Warning: LIDAR services initialization failed: {e}")
    lidar_handler = None
    point_cloud_processor = None
    geological_extractor = None

# Initialize auto-training services
if AUTO_TRAINING_AVAILABLE:
    try:
        # Try to load existing models
        auto_trainer.load_trained_models()
        print("[OK] Auto-training services initialized")
        
        # Check if models exist, if not, generate and train with sample data
        model_status = auto_trainer.get_model_status()
        if not any([
            model_status['traditional_models']['classifier_file_exists'],
            model_status['traditional_models']['regressor_file_exists'],
            model_status['lstm_model']['model_file_exists']
        ]):
            print("🤖 No pre-trained models found. Training with sample data...")
            try:
                # Generate training data and train models automatically
                training_data = auto_trainer.generate_comprehensive_training_data(n_samples=5000)
                auto_trainer.train_traditional_models(training_data)
                print("✅ Traditional models trained successfully")
                
                # Train LSTM if TensorFlow is available
                if model_status['lstm_model']['available']:
                    auto_trainer.train_lstm_model(training_data)
                    print("✅ LSTM model trained successfully")
                
            except Exception as train_error:
                print(f"⚠️ Warning: Auto-training during startup failed: {train_error}")
        else:
            print("✅ Pre-trained models loaded successfully")
            
    except Exception as e:
        print(f"⚠️ Warning: Auto-training initialization failed: {e}")
else:
    print("⚠️ Warning: Auto-training not available")
    dl_model = None

def send_email(to_email, subject, body, attachment_path=None):
    """Send email with optional attachment"""
    try:
        # Using SMTP with Gmail (can be configured for other providers)
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email = os.getenv('SMTP_EMAIL', 'your_email@gmail.com')
        sender_password = os.getenv('SMTP_PASSWORD', 'your_password')
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'html'))
        
        # Add attachment if provided
        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {os.path.basename(attachment_path)}',
            )
            msg.attach(part)
        
        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, to_email, text)
        server.quit()
        
        return True
    except Exception as e:
        print(f"Email sending failed: {e}")
        return False

def generate_risk_analysis_report(user, report_type, report_id):
    """Generate PDF risk analysis report and email it to user"""
    try:
        # Get recent risk assessments
        assessments = RiskAssessment.query.order_by(RiskAssessment.timestamp.desc()).limit(100).all()
        
        # Get recent sensor data
        sensor_data = SensorData.query.order_by(SensorData.timestamp.desc()).limit(500).all()
        
        # Create PDF
        filename = f"risk_analysis_report_{report_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join('reports', filename)
        
        # Create reports directory if it doesn't exist
        os.makedirs('reports', exist_ok=True)
        
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        story = []
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        heading_style = styles['Heading1']
        normal_style = styles['Normal']
        
        # Title
        story.append(Paragraph(f"Rockfall Risk Analysis Report", title_style))
        story.append(Spacer(1, 12))
        
        # Report info
        story.append(Paragraph(f"Generated for: {user.first_name} {user.last_name}", normal_style))
        story.append(Paragraph(f"Company: {user.company}", normal_style))
        story.append(Paragraph(f"Report Type: {report_type.title()}", normal_style))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}", normal_style))
        story.append(Spacer(1, 20))
        
        # Risk Summary
        if assessments:
            latest_assessment = assessments[0]
            story.append(Paragraph("Current Risk Status", heading_style))
            story.append(Paragraph(f"Risk Level: <b>{latest_assessment.risk_level}</b>", normal_style))
            story.append(Paragraph(f"Risk Probability: <b>{latest_assessment.risk_probability:.2f}%</b>", normal_style))
            story.append(Paragraph(f"Last Updated: {latest_assessment.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}", normal_style))
            story.append(Spacer(1, 12))
            
            # Risk trend analysis
            high_risk_count = len([a for a in assessments[:24] if a.risk_level in ['HIGH', 'CRITICAL']])
            story.append(Paragraph(f"High Risk Incidents (Last 24 assessments): <b>{high_risk_count}</b>", normal_style))
            story.append(Spacer(1, 20))
        
        # Sensor Data Summary
        story.append(Paragraph("Sensor Data Summary", heading_style))
        if sensor_data:
            # Group by sensor type
            sensor_types = {}
            for data in sensor_data[:50]:  # Last 50 readings
                if data.sensor_type not in sensor_types:
                    sensor_types[data.sensor_type] = []
                sensor_types[data.sensor_type].append(data.value)
            
            for sensor_type, values in sensor_types.items():
                avg_value = sum(values) / len(values)
                max_value = max(values)
                min_value = min(values)
                
                story.append(Paragraph(f"<b>{sensor_type.title()}:</b>", normal_style))
                story.append(Paragraph(f"  Average: {avg_value:.2f}", normal_style))
                story.append(Paragraph(f"  Range: {min_value:.2f} - {max_value:.2f}", normal_style))
                story.append(Spacer(1, 6))
        
        story.append(Spacer(1, 20))
        
        # Recommendations
        story.append(Paragraph("Recommendations", heading_style))
        if assessments and assessments[0].risk_level == 'CRITICAL':
            story.append(Paragraph("• <b>IMMEDIATE ACTION REQUIRED:</b> Critical risk detected", normal_style))
            story.append(Paragraph("• Evacuate personnel from high-risk areas immediately", normal_style))
            story.append(Paragraph("• Contact emergency response team", normal_style))
        elif assessments and assessments[0].risk_level == 'HIGH':
            story.append(Paragraph("• Increase monitoring frequency", normal_style))
            story.append(Paragraph("• Restrict access to high-risk areas", normal_style))
            story.append(Paragraph("• Prepare emergency response procedures", normal_style))
        else:
            story.append(Paragraph("• Continue regular monitoring", normal_style))
            story.append(Paragraph("• Maintain current safety protocols", normal_style))
            story.append(Paragraph("• Schedule routine equipment checks", normal_style))
        
        story.append(Spacer(1, 20))
        story.append(Paragraph("This report was automatically generated by the AI Rockfall Prediction System.", normal_style))
        
        # Build PDF
        doc.build(story)
        
        # Send email
        subject = f"Rockfall Risk Analysis Report - {report_type.title()}"
        body = f"""
        <html>
        <body>
        <h2>Rockfall Risk Analysis Report</h2>
        <p>Dear {user.first_name} {user.last_name},</p>
        <p>Please find attached your {report_type} rockfall risk analysis report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}.</p>
        
        <h3>Report Summary:</h3>
        <ul>
        """
        
        if assessments:
            body += f"<li><strong>Current Risk Level:</strong> {assessments[0].risk_level}</li>"
            body += f"<li><strong>Risk Probability:</strong> {assessments[0].risk_probability:.2f}%</li>"
        
        body += f"<li><strong>Total Sensor Readings:</strong> {len(sensor_data)} recent readings analyzed</li>"
        body += f"<li><strong>Total Risk Assessments:</strong> {len(assessments)} assessments reviewed</li>"
        
        body += """
        </ul>
        
        <p>Please review the attached detailed report and take appropriate actions based on the recommendations provided.</p>
        
        <p>Best regards,<br>
        AI Rockfall Prediction System</p>
        </body>
        </html>
        """
        
        success = send_email(user.email, subject, body, filepath)
        
        # Clean up file after sending (optional)
        try:
            if success:
                os.remove(filepath)
        except:
            pass
        
        return success
        
    except Exception as e:
        print(f"Report generation failed: {e}")
        return False

import re
import dns.resolver
from urllib.parse import urlparse

def validate_email_domain(email):
    """Validate email format and domain existence"""
    try:
        # Basic email format validation
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_regex, email):
            return False, "Invalid email format"
        
        # Extract domain
        domain = email.split('@')[1]
        
        # Check if domain has MX record (email capability)
        try:
            resolver = dns.resolver.Resolver()
            resolver.lifetime = 2.0  # 2 second timeout
            mx_records = resolver.resolve(domain, 'MX')
            if not mx_records:
                return False, "Email domain does not accept emails"
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, Exception):
            # If MX check fails, try A record as fallback
            try:
                resolver = dns.resolver.Resolver()
                resolver.lifetime = 2.0
                resolver.resolve(domain, 'A')
            except Exception:
                # In development/test environments or if DNS is restricted, allow valid format
                return True, "Valid email format (offline/local fallback)"
        
        return True, "Valid email domain"
    except Exception as e:
        return False, f"Email validation error: {str(e)}"

def validate_password_strength(password):
    """Validate password meets security requirements"""
    errors = []
    
    # Minimum length
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    
    # Maximum length (prevent DoS attacks)
    if len(password) > 128:
        errors.append("Password must be less than 128 characters")
    
    # Check for uppercase letter
    if not re.search(r'[A-Z]', password):
        errors.append("Password must contain at least one uppercase letter")
    
    # Check for lowercase letter
    if not re.search(r'[a-z]', password):
        errors.append("Password must contain at least one lowercase letter")
    
    # Check for digit
    if not re.search(r'\d', password):
        errors.append("Password must contain at least one number")
    
    # Check for special character
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        errors.append("Password must contain at least one special character (!@#$%^&*(),.?\":{}|<>)")
    
    # Check for common weak patterns
    weak_patterns = [
        r'123456',
        r'password',
        r'qwerty',
        r'abc123',
        r'admin',
        r'letmein'
    ]
    
    for pattern in weak_patterns:
        if re.search(pattern, password.lower()):
            errors.append("Password contains common weak patterns")
            break
    
    return len(errors) == 0, errors

def validate_name(name, field_name):
    """Validate name fields"""
    if not name or len(name.strip()) < 2:
        return False, f"{field_name} must be at least 2 characters long"
    
    if len(name) > 50:
        return False, f"{field_name} must be less than 50 characters"
    
    # Allow letters, spaces, hyphens, apostrophes
    if not re.match(r"^[a-zA-Z\s\-']+$", name):
        return False, f"{field_name} can only contain letters, spaces, hyphens, and apostrophes"
    
    return True, "Valid name"

# Import and register data sources blueprint
try:
    from data_sources_api import data_sources_bp
    app.register_blueprint(data_sources_bp)
    print("[OK] Data sources API registered")
except Exception as e:
    print(f"⚠️ Warning: Data sources API registration failed: {e}")

# Authentication Routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user with enhanced validation"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['email', 'password', 'first_name', 'last_name']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'{field} is required'}), 400
        
        # Validate email format and domain
        email_valid, email_message = validate_email_domain(data['email'])
        if not email_valid:
            return jsonify({'error': email_message}), 400
        
        # Validate password strength
        password_valid, password_errors = validate_password_strength(data['password'])
        if not password_valid:
            return jsonify({
                'error': 'Password does not meet security requirements',
                'details': password_errors
            }), 400
        
        # Validate first name
        first_name_valid, first_name_message = validate_name(data['first_name'], 'First name')
        if not first_name_valid:
            return jsonify({'error': first_name_message}), 400
        
        # Validate last name
        last_name_valid, last_name_message = validate_name(data['last_name'], 'Last name')
        if not last_name_valid:
            return jsonify({'error': last_name_message}), 400
        
        # Validate company name if provided
        if data.get('company'):
            if len(data['company']) > 100:
                return jsonify({'error': 'Company name must be less than 100 characters'}), 400
            if not re.match(r"^[a-zA-Z0-9\s\-.,&()]+$", data['company']):
                return jsonify({'error': 'Company name contains invalid characters'}), 400
        
        # Validate role
        valid_roles = ['user', 'admin', 'engineer', 'supervisor']
        role = data.get('role', 'user').lower()
        if role not in valid_roles:
            return jsonify({'error': f'Invalid role. Must be one of: {", ".join(valid_roles)}'}), 400
        
        # Check if user already exists
        existing_user = User.query.filter_by(email=data['email'].lower()).first()
        if existing_user:
            return jsonify({'error': 'Email already registered'}), 400
        
        # Create new user
        user = User(
            email=data['email'].lower().strip(),
            first_name=data['first_name'].strip().title(),
            last_name=data['last_name'].strip().title(),
            company=data.get('company', '').strip(),
            role=role
        )
        user.set_password(data['password'])
        
        db.session.add(user)
        db.session.commit()
        
        return jsonify({
            'message': 'User registered successfully',
            'user': user.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Authenticate user and return JWT token"""
    try:
        data = request.json
        
        if not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Email and password are required'}), 400
        
        # Find user
        user = User.query.filter_by(email=data['email']).first()
        
        if not user or not user.check_password(data['password']):
            return jsonify({'error': 'Invalid email or password'}), 401
        
        if not user.is_active:
            return jsonify({'error': 'Account is deactivated'}), 401
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        # Generate JWT token
        access_token = create_access_token(identity=str(user.id))
        
        return jsonify({
            'message': 'Login successful',
            'access_token': access_token,
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """Get current user profile"""
    try:
        user_id = int(get_jwt_identity())
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/logout', methods=['POST'])
@jwt_required()
def logout():
    """Logout user (client should delete token)"""
    return jsonify({'message': 'Logout successful'}), 200

# Report Generation Routes
@app.route('/api/reports/generate', methods=['POST'])
@jwt_required()
def generate_report():
    """Generate and email risk analysis report"""
    try:
        user_id = int(get_jwt_identity())
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.json
        report_type = data.get('report_type', 'daily')
        
        # Create report request
        report_request = ReportRequest(
            user_id=user_id,
            report_type=report_type,
            status='generating'
        )
        db.session.add(report_request)
        db.session.commit()
        
        # Generate report in background (simplified for demo)
        success = generate_risk_analysis_report(user, report_type, report_request.id)
        
        if success:
            report_request.status = 'sent'
            report_request.email_sent = True
            report_request.generated_at = datetime.utcnow()
        else:
            report_request.status = 'failed'
        
        db.session.commit()
        
        return jsonify({
            'message': 'Report generation requested',
            'report_id': report_request.id,
            'status': report_request.status
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reports/history', methods=['GET'])
@jwt_required()
def get_report_history():
    """Get user's report history"""
    try:
        user_id = int(get_jwt_identity())
        
        reports = ReportRequest.query.filter_by(user_id=user_id)\
                                   .order_by(ReportRequest.requested_at.desc())\
                                   .limit(20).all()
        
        report_list = []
        for report in reports:
            report_list.append({
                'id': report.id,
                'report_type': report.report_type,
                'status': report.status,
                'email_sent': report.email_sent,
                'requested_at': report.requested_at.isoformat(),
                'generated_at': report.generated_at.isoformat() if report.generated_at else None
            })
        
        return jsonify({
            'reports': report_list
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sensor-data', methods=['POST'])
def receive_sensor_data():
    """Receive and store sensor data"""
    try:
        data = request.json
        sensor_data = SensorData(
            sensor_id=data['sensor_id'],
            sensor_type=data['sensor_type'],
            location_x=data['location_x'],
            location_y=data['location_y'],
            value=data['value'],
            unit=data['unit'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )
        db.session.add(sensor_data)
        db.session.commit()
        
        # Trigger prediction if enough data
        if should_trigger_prediction():
            try:
                prediction_result = predictor.predict_rockfall_risk()
                process_prediction_result(prediction_result)
            except Exception as e:
                print(f"Prediction error: {e}")
        
        return jsonify({'status': 'success', 'id': sensor_data.id})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/risk-assessment', methods=['GET'])
def get_risk_assessment():
    """Get current risk assessment including LIDAR analysis"""
    try:
        # Get latest traditional risk assessment
        latest_assessment = RiskAssessment.query.order_by(RiskAssessment.timestamp.desc()).first()
        
        # Get recent LIDAR analyses
        recent_lidar_analyses = LIDARAnalysis.query.filter(
            LIDARAnalysis.timestamp >= datetime.utcnow() - timedelta(days=7)
        ).order_by(LIDARAnalysis.timestamp.desc()).limit(5).all()
        
        # Combine traditional and LIDAR assessments
        combined_assessment = {
            'traditional_assessment': None,
            'lidar_assessment': None,
            'combined_risk_level': 'UNKNOWN',
            'confidence': 0.0,
            'recommendations': [],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Add traditional assessment
        if latest_assessment:
            combined_assessment['traditional_assessment'] = {
                'risk_level': latest_assessment.risk_level,
                'probability': latest_assessment.probability,
                'affected_zones': json.loads(latest_assessment.affected_zones),
                'timestamp': latest_assessment.timestamp.isoformat(),
                'recommendations': json.loads(latest_assessment.recommendations) if latest_assessment.recommendations else []
            }
        
        # Add LIDAR assessment
        if recent_lidar_analyses:
            latest_lidar = recent_lidar_analyses[0]
            
            # Calculate average stability from recent analyses
            avg_stability = sum(analysis.stability_score for analysis in recent_lidar_analyses if analysis.stability_score) / len(recent_lidar_analyses)
            
            # Get risk level distribution
            risk_levels = [analysis.risk_level for analysis in recent_lidar_analyses if analysis.risk_level]
            risk_distribution = {}
            for level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
                risk_distribution[level] = risk_levels.count(level)
            
            combined_assessment['lidar_assessment'] = {
                'latest_analysis_id': latest_lidar.id,
                'average_stability_score': avg_stability,
                'risk_level': latest_lidar.risk_level,
                'timestamp': latest_lidar.timestamp.isoformat(),
                'risk_distribution': risk_distribution,
                'number_of_analyses': len(recent_lidar_analyses)
            }
            
            # Enhanced recommendations based on LIDAR data
            lidar_results = json.loads(latest_lidar.results_json)
            geological_analysis = lidar_results.get('geological_analysis', {})
            
            lidar_recommendations = []
            
            # Check for specific geological hazards
            if geological_analysis.get('risk_factors', {}).get('high_slope_areas'):
                lidar_recommendations.append("High slope areas detected - implement additional monitoring")
            
            if geological_analysis.get('risk_factors', {}).get('major_discontinuities'):
                lidar_recommendations.append("Major discontinuities identified - geological assessment recommended")
            
            if geological_analysis.get('risk_factors', {}).get('active_weathering'):
                lidar_recommendations.append("Active weathering detected - monitor degradation patterns")
            
            if geological_analysis.get('risk_factors', {}).get('crack_networks'):
                lidar_recommendations.append("Crack networks observed - structural analysis needed")
            
            if geological_analysis.get('risk_factors', {}).get('unstable_overhangs'):
                lidar_recommendations.append("Unstable overhangs detected - immediate safety assessment required")
            
            combined_assessment['lidar_assessment']['recommendations'] = lidar_recommendations
        
        # Calculate combined risk level
        traditional_risk = latest_assessment.risk_level if latest_assessment else 'MEDIUM'
        lidar_risk = recent_lidar_analyses[0].risk_level if recent_lidar_analyses else 'MEDIUM'
        
        # Risk level priority mapping
        risk_priority = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
        
        # Take the higher risk level
        traditional_priority = risk_priority.get(traditional_risk, 2)
        lidar_priority = risk_priority.get(lidar_risk, 2)
        
        combined_priority = max(traditional_priority, lidar_priority)
        
        # Add weight if both assessments agree on high risk
        if traditional_priority >= 3 and lidar_priority >= 3:
            combined_priority = min(4, combined_priority + 0.5)
        
        # Map back to risk level
        priority_to_risk = {1: 'LOW', 2: 'MEDIUM', 3: 'HIGH', 4: 'CRITICAL'}
        combined_assessment['combined_risk_level'] = priority_to_risk[int(combined_priority)]
        
        # Calculate confidence based on data availability
        confidence = 0.5  # Base confidence
        if latest_assessment:
            confidence += 0.25
        if recent_lidar_analyses:
            confidence += 0.25
        if len(recent_lidar_analyses) > 2:  # Multiple LIDAR analyses increase confidence
            confidence += 0.1
        
        combined_assessment['confidence'] = min(1.0, confidence)
        
        # Combine recommendations
        all_recommendations = []
        if latest_assessment and latest_assessment.recommendations:
            all_recommendations.extend(json.loads(latest_assessment.recommendations))
        if recent_lidar_analyses:
            all_recommendations.extend(combined_assessment['lidar_assessment']['recommendations'])
        
        combined_assessment['recommendations'] = list(set(all_recommendations))  # Remove duplicates
        
        return jsonify(combined_assessment)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/risk-map', methods=['GET'])
def get_risk_map():
    """Get risk map data for visualization including LIDAR zones"""
    try:
        # Get recent sensor data for traditional risk mapping
        recent_data = SensorData.query.filter(
            SensorData.timestamp >= datetime.utcnow() - timedelta(hours=24)
        ).all()
        
        traditional_risk_zones = predictor.generate_risk_map(recent_data)
        
        # Get LIDAR risk zones
        lidar_risk_zones = []
        recent_lidar_analyses = LIDARAnalysis.query.filter(
            LIDARAnalysis.timestamp >= datetime.utcnow() - timedelta(days=7)
        ).all()
        
        for analysis in recent_lidar_analyses:
            scan = analysis.scan
            if scan.bounds_json:
                bounds = json.loads(scan.bounds_json)
                
                # Create risk zone from scan bounds
                center_x = (bounds['x'][0] + bounds['x'][1]) / 2
                center_y = (bounds['y'][0] + bounds['y'][1]) / 2
                
                # Convert to approximate lat/lng (this would need proper coordinate transformation in production)
                lat = center_y / 111000  # Very rough approximation
                lng = center_x / 111000
                
                risk_value = 1.0 - analysis.stability_score if analysis.stability_score else 0.5
                
                lidar_risk_zones.append({
                    'lat': lat,
                    'lng': lng,
                    'risk_value': risk_value,
                    'risk_level': analysis.risk_level,
                    'source': 'LIDAR',
                    'scan_id': scan.id,
                    'analysis_id': analysis.id,
                    'timestamp': analysis.timestamp.isoformat(),
                    'stability_score': analysis.stability_score
                })
        
        # Mark traditional zones
        for zone in traditional_risk_zones:
            zone['source'] = 'SENSOR'
        
        # Combine all risk zones
        all_risk_zones = traditional_risk_zones + lidar_risk_zones
        
        return jsonify({
            'risk_zones': all_risk_zones,
            'traditional_zones': len(traditional_risk_zones),
            'lidar_zones': len(lidar_risk_zones),
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get recent alerts"""
    try:
        alerts = Alert.query.filter(
            Alert.timestamp >= datetime.utcnow() - timedelta(days=7)
        ).order_by(Alert.timestamp.desc()).all()
        
        alert_list = []
        for alert in alerts:
            alert_list.append({
                'id': alert.id,
                'alert_type': alert.alert_type,
                'severity': alert.severity,
                'message': alert.message,
                'location': alert.location,
                'timestamp': alert.timestamp.isoformat(),
                'status': alert.status
            })
        
        return jsonify({'alerts': alert_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    """Get rockfall probability forecast"""
    try:
        forecast_data = predictor.generate_forecast()
        return jsonify(forecast_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== ENHANCED DASHBOARD API ENDPOINTS ====================

@app.route('/api/enhanced-dashboard/statistics', methods=['GET'])
def get_mining_statistics():
    """Get comprehensive mining statistics for enhanced dashboard"""
    try:
        # Calculate time periods
        now = datetime.utcnow()
        last_24h = now - timedelta(hours=24)
        last_week = now - timedelta(days=7)
        last_month = now - timedelta(days=30)
        
        # Count active sensors
        active_sensors = SensorData.query.filter(
            SensorData.timestamp >= last_24h
        ).distinct(SensorData.sensor_id).count()
        
        # Count alerts by severity
        recent_alerts = Alert.query.filter(Alert.timestamp >= last_week).all()
        alert_counts = {
            'total': len(recent_alerts),
            'critical': len([a for a in recent_alerts if a.severity == 'CRITICAL']),
            'high': len([a for a in recent_alerts if a.severity == 'HIGH']),
            'medium': len([a for a in recent_alerts if a.severity == 'MEDIUM']),
            'low': len([a for a in recent_alerts if a.severity == 'LOW'])
        }
        
        # Get current risk assessment
        current_risk = RiskAssessment.query.order_by(RiskAssessment.timestamp.desc()).first()
        
        # Calculate risk trend (simplified)
        recent_risks = RiskAssessment.query.filter(
            RiskAssessment.timestamp >= last_24h
        ).order_by(RiskAssessment.timestamp.desc()).limit(10).all()
        
        risk_trend = "stable"
        if len(recent_risks) >= 2:
            latest_prob = recent_risks[0].probability
            previous_prob = sum([r.probability for r in recent_risks[1:]]) / (len(recent_risks) - 1)
            if latest_prob > previous_prob + 0.1:
                risk_trend = "increasing"
            elif latest_prob < previous_prob - 0.1:
                risk_trend = "decreasing"
        
        # Get sensor data trend
        sensor_readings = SensorData.query.filter(
            SensorData.timestamp >= last_24h
        ).count()
        
        # Calculate system uptime (mock data)
        uptime_percentage = round(random.uniform(95, 99.5), 1)
        
        statistics = {
            'active_sensors': active_sensors,
            'total_alerts': alert_counts['total'],
            'critical_alerts': alert_counts['critical'],
            'current_risk_level': current_risk.risk_level if current_risk else 'LOW',
            'current_probability': round(current_risk.probability, 2) if current_risk else 0.25,
            'risk_trend': risk_trend,
            'sensor_readings_24h': sensor_readings,
            'system_uptime': uptime_percentage,
            'alert_distribution': {
                'critical': alert_counts['critical'],
                'high': alert_counts['high'],
                'medium': alert_counts['medium'],
                'low': alert_counts['low']
            },
            'mining_zones': {
                'total': 8,
                'active': 6,
                'under_maintenance': 1,
                'high_risk': 1
            },
            'production_status': {
                'operational': True,
                'current_shift': "Day Shift",
                'workers_on_site': random.randint(45, 85)
            }
        }
        
        return jsonify(statistics)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/enhanced-dashboard/india-mining-data', methods=['GET'])
def get_india_mining_data():
    """Get India-specific mining risk data for map visualization"""
    try:
        # Mock data for major mining states in India
        india_mining_data = {
            'states': [
                {
                    'name': 'Odisha',
                    'risk_level': 'MEDIUM',
                    'active_mines': 156,
                    'incidents_last_month': 3,
                    'coordinates': [20.9517, 85.0985]
                },
                {
                    'name': 'Chhattisgarh',
                    'risk_level': 'HIGH',
                    'active_mines': 89,
                    'incidents_last_month': 7,
                    'coordinates': [21.2787, 81.8661]
                },
                {
                    'name': 'Jharkhand',
                    'risk_level': 'MEDIUM',
                    'active_mines': 67,
                    'incidents_last_month': 2,
                    'coordinates': [23.6102, 85.2799]
                },
                {
                    'name': 'West Bengal',
                    'risk_level': 'LOW',
                    'active_mines': 34,
                    'incidents_last_month': 1,
                    'coordinates': [22.9868, 87.8550]
                },
                {
                    'name': 'Rajasthan',
                    'risk_level': 'MEDIUM',
                    'active_mines': 45,
                    'incidents_last_month': 2,
                    'coordinates': [27.0238, 74.2179]
                },
                {
                    'name': 'Madhya Pradesh',
                    'risk_level': 'LOW',
                    'active_mines': 28,
                    'incidents_last_month': 0,
                    'coordinates': [22.9734, 78.6569]
                }
            ],
            'national_statistics': {
                'total_mines': 419,
                'total_incidents_month': 15,
                'average_risk_score': 6.2,
                'mines_by_risk': {
                    'low': 187,
                    'medium': 189,
                    'high': 43
                }
            },
            'recent_incidents': [
                {
                    'location': 'Odisha',
                    'type': 'Minor rockfall',
                    'date': '2024-01-15',
                    'severity': 'LOW'
                },
                {
                    'location': 'Chhattisgarh',
                    'type': 'Slope instability',
                    'date': '2024-01-14',
                    'severity': 'MEDIUM'
                },
                {
                    'location': 'Jharkhand',
                    'type': 'Equipment damage',
                    'date': '2024-01-12',
                    'severity': 'LOW'
                }
            ]
        }
        
        return jsonify(india_mining_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/enhanced-dashboard/risk-distribution', methods=['GET'])
def get_risk_distribution():
    """Get risk distribution data for charts"""
    try:
        # Get recent risk assessments
        recent_risks = RiskAssessment.query.filter(
            RiskAssessment.timestamp >= datetime.utcnow() - timedelta(days=30)
        ).all()
        
        # Count by risk level
        risk_counts = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
        for risk in recent_risks:
            if risk.risk_level in risk_counts:
                risk_counts[risk.risk_level] += 1
        
        # Generate time series data for last 7 days
        time_series = []
        for i in range(7):
            date = datetime.utcnow() - timedelta(days=6-i)
            # Mock data - replace with actual database queries
            time_series.append({
                'date': date.strftime('%Y-%m-%d'),
                'risk_score': round(random.uniform(2, 8), 1),
                'incidents': random.randint(0, 3),
                'sensor_alerts': random.randint(0, 10)
            })
        
        # Sensor type distribution
        sensor_types = SensorData.query.with_entities(
            SensorData.sensor_type,
            db.func.count(SensorData.id).label('count')
        ).filter(
            SensorData.timestamp >= datetime.utcnow() - timedelta(hours=24)
        ).group_by(SensorData.sensor_type).all()
        
        sensor_distribution = {sensor_type: count for sensor_type, count in sensor_types}
        
        return jsonify({
            'risk_distribution': risk_counts,
            'time_series': time_series,
            'sensor_distribution': sensor_distribution
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== END ENHANCED DASHBOARD API ENDPOINTS ====================

@app.route('/api/test-alert', methods=['POST'])
def send_test_alert():
    """Send a test alert to configured contacts"""
    try:
        # Load emergency contacts
        import json
        with open('config/emergency_contacts.json', 'r') as f:
            config = json.load(f)
            contacts = config['emergency_contacts']
        
        # Create test alert
        test_alert = {
            'alert_type': 'SYSTEM_TEST',
            'severity': 'HIGH',
            'message': f'Test alert from Rockfall Prediction System - {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")}',
            'location': 'Test Zone',
            'probability': 0.75,
            'timestamp': datetime.utcnow().isoformat(),
            'recommendations': [
                'This is a test alert to verify the notification system',
                'No action required - system test only',
                'Check dashboard for system status'
            ]
        }
        
        # Format messages
        sms_message = f"""🚨 TEST ALERT 🚨
System: Rockfall Prediction
Severity: {test_alert['severity']}
Time: {datetime.utcnow().strftime('%H:%M')}

{test_alert['message']}

This is a TEST - No action required."""
        
        email_subject = f"🧪 TEST ALERT - Rockfall Prediction System"
        email_message = f"""TEST ALERT - ROCKFALL PREDICTION SYSTEM

This is a test message to verify the alert system is working correctly.

Alert Details:
- Type: {test_alert['alert_type']}
- Severity: {test_alert['severity']}
- Time: {test_alert['timestamp']}
- Message: {test_alert['message']}

System Status: Operational
Dashboard: http://localhost:3000

This is an automated test from the AI-Based Rockfall Prediction System.
No action is required - this is for testing purposes only."""
        
        # Send alerts (mock implementation)
        sent_alerts = []
        for contact in contacts:
            contact_alerts = {
                'name': contact['name'],
                'role': contact['role'],
                'alerts_sent': []
            }
            
            if contact.get('phone'):
                # Mock SMS sending
                contact_alerts['alerts_sent'].append({
                    'type': 'SMS',
                    'destination': contact['phone'],
                    'status': 'sent_mock',
                    'message': sms_message[:100] + '...'
                })
            
            if contact.get('email'):
                # Mock email sending
                contact_alerts['alerts_sent'].append({
                    'type': 'EMAIL',
                    'destination': contact['email'],
                    'status': 'sent_mock',
                    'subject': email_subject
                })
            
            sent_alerts.append(contact_alerts)
        
        return jsonify({
            'status': 'success',
            'message': 'Test alerts sent successfully',
            'alert_details': test_alert,
            'contacts_notified': len(contacts),
            'sent_alerts': sent_alerts,
            'note': 'This is a mock implementation. Configure Twilio and SendGrid for real alerts.'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to send test alert: {str(e)}'}), 500

def should_trigger_prediction():
    """Determine if prediction should be triggered based on data availability"""
    recent_count = SensorData.query.filter(
        SensorData.timestamp >= datetime.utcnow() - timedelta(minutes=30)
    ).count()
    return recent_count >= 10  # Trigger if we have at least 10 recent readings

def process_prediction_result(prediction_result):
    """Process prediction result and generate alerts if necessary"""
    risk_level = prediction_result['risk_level']
    probability = prediction_result['probability']
    
    try:
        # Store risk assessment
        assessment = RiskAssessment(
            risk_level=risk_level,
            probability=probability,
            affected_zones=json.dumps(prediction_result['affected_zones']),
            recommendations=json.dumps(prediction_result['recommendations'])  # Convert to JSON string
        )
        db.session.add(assessment)
        
        # Generate alert if high risk
        if risk_level in ['HIGH', 'CRITICAL']:
            alert = Alert(
                alert_type='ROCKFALL_WARNING',
                severity=risk_level,
                message=f"High rockfall risk detected. Probability: {probability:.2%}",
                location=json.dumps(prediction_result['affected_zones'])
            )
            db.session.add(alert)
            
            # Send notifications (mock for now)
            print(f"🚨 Alert generated: {alert.message}")
        
        db.session.commit()
        print(f"[OK] Risk assessment stored: {risk_level} ({probability:.1%})")
        
    except Exception as e:
        print(f"Database error: {e}")
        db.session.rollback()

# ==================== LIDAR API ENDPOINTS ====================

# Configuration for file uploads
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), '..', 'uploads', 'lidar')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/api/lidar/upload', methods=['POST'])
def upload_lidar_file():
    """Upload LIDAR point cloud file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        filename = secure_filename(file.filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        if lidar_handler is None:
            return jsonify({'error': 'LIDAR processing not available'}), 503
        
        if file_ext not in lidar_handler.get_supported_formats():
            return jsonify({
                'error': f'Unsupported file format: {file_ext}',
                'supported_formats': lidar_handler.get_supported_formats()
            }), 400
        
        # Save file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Validate file
        if not lidar_handler.validate_file(file_path):
            os.remove(file_path)
            return jsonify({'error': 'Invalid or corrupted file'}), 400
        
        # Load and extract metadata
        try:
            pcd, metadata = lidar_handler.load_point_cloud(file_path)
            
            # Store in database
            scan = LIDARScan(
                filename=metadata.filename,
                file_path=file_path,
                file_format=metadata.format,
                num_points=metadata.num_points,
                bounds_json=json.dumps(metadata.bounds),
                scan_date=metadata.scan_date,
                coordinate_system=metadata.coordinate_system,
                scanner_info=json.dumps(metadata.scanner_info) if metadata.scanner_info else None,
                processing_status='UPLOADED'
            )
            
            db.session.add(scan)
            db.session.commit()
            
            return jsonify({
                'scan_id': scan.id,
                'filename': filename,
                'num_points': metadata.num_points,
                'bounds': metadata.bounds,
                'format': metadata.format,
                'status': 'uploaded'
            })
            
        except Exception as e:
            os.remove(file_path)
            return jsonify({'error': f'Failed to process file: {str(e)}'}), 500
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/lidar/scans', methods=['GET'])
def list_lidar_scans():
    """List all LIDAR scans"""
    try:
        scans = LIDARScan.query.order_by(LIDARScan.upload_date.desc()).all()
        
        scan_list = []
        for scan in scans:
            scan_data = {
                'id': scan.id,
                'filename': scan.filename,
                'format': scan.file_format,
                'num_points': scan.num_points,
                'upload_date': scan.upload_date.isoformat(),
                'processing_status': scan.processing_status,
                'bounds': json.loads(scan.bounds_json) if scan.bounds_json else None
            }
            
            if scan.scan_date:
                scan_data['scan_date'] = scan.scan_date.isoformat()
            
            scan_list.append(scan_data)
        
        return jsonify({'scans': scan_list})
        
    except Exception as e:
        return jsonify({'error': f'Failed to list scans: {str(e)}'}), 500

@app.route('/api/lidar/analyze/<int:scan_id>', methods=['POST'])
def analyze_lidar_scan(scan_id):
    """Analyze LIDAR scan for geological features and risk assessment"""
    try:
        # Get scan from database
        scan = LIDARScan.query.get_or_404(scan_id)
        
        if not os.path.exists(scan.file_path):
            return jsonify({'error': 'Scan file not found'}), 404
        
        if lidar_handler is None or geological_extractor is None or dl_model is None:
            return jsonify({'error': 'LIDAR processing not available'}), 503
        
        # Update status
        scan.processing_status = 'PROCESSING'
        db.session.commit()
        
        start_time = time.time()
        
        # Load point cloud
        pcd, metadata = lidar_handler.load_point_cloud(scan.file_path)
        
        # Preprocess point cloud
        pcd_processed = point_cloud_processor.preprocess(pcd)
        
        # Extract geological features
        geological_results = geological_extractor.analyze_point_cloud(pcd_processed)
        
        # Deep learning risk assessment
        dl_risk = dl_model.predict_risk(pcd_processed)
        dl_features = dl_model.detect_geological_features(pcd_processed)
        
        # Compute point cloud statistics
        stats = point_cloud_processor.compute_point_cloud_statistics(pcd_processed)
        
        processing_time = time.time() - start_time
        
        # Prepare analysis results
        analysis_results = {
            'geological_analysis': {
                'stability_score': geological_results.stability_score,
                'slope_angles': {
                    'mean': float(np.mean(geological_results.slope_angles)) if len(geological_results.slope_angles) > 0 else 0,
                    'max': float(np.max(geological_results.slope_angles)) if len(geological_results.slope_angles) > 0 else 0,
                    'std': float(np.std(geological_results.slope_angles)) if len(geological_results.slope_angles) > 0 else 0
                },
                'surface_roughness': {
                    'mean': float(np.mean(geological_results.surface_roughness)) if len(geological_results.surface_roughness) > 0 else 0,
                    'max': float(np.max(geological_results.surface_roughness)) if len(geological_results.surface_roughness) > 0 else 0
                },
                'discontinuities': geological_results.discontinuity_planes,
                'cracks': geological_results.crack_features,
                'overhangs': geological_results.overhang_regions,
                'weathering': geological_results.weathering_indicators,
                'risk_factors': geological_results.risk_factors
            },
            'deep_learning_analysis': {
                'risk_assessment': dl_risk,
                'detected_features': dl_features
            },
            'point_cloud_statistics': stats,
            'processing_info': {
                'processing_time': processing_time,
                'original_points': metadata.num_points,
                'processed_points': len(pcd_processed.points),
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        
        # Determine overall risk level
        combined_risk_score = (geological_results.stability_score * 0.6 + 
                             dl_risk['confidence'] * 0.4)
        
        if combined_risk_score < 0.3:
            overall_risk = 'CRITICAL'
        elif combined_risk_score < 0.5:
            overall_risk = 'HIGH'
        elif combined_risk_score < 0.7:
            overall_risk = 'MEDIUM'
        else:
            overall_risk = 'LOW'
        
        # Store analysis results
        analysis = LIDARAnalysis(
            scan_id=scan.id,
            analysis_type='comprehensive',
            results_json=json.dumps(analysis_results),
            stability_score=geological_results.stability_score,
            risk_level=overall_risk,
            processing_time=processing_time
        )
        
        db.session.add(analysis)
        
        # Update scan status
        scan.processing_status = 'COMPLETED'
        db.session.commit()
        
        return jsonify({
            'analysis_id': analysis.id,
            'scan_id': scan.id,
            'risk_level': overall_risk,
            'stability_score': geological_results.stability_score,
            'processing_time': processing_time,
            'results': analysis_results
        })
        
    except Exception as e:
        # Update scan status to failed
        try:
            scan = LIDARScan.query.get(scan_id)
            if scan:
                scan.processing_status = 'FAILED'
                db.session.commit()
        except:
            pass
        
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/lidar/analysis/<int:analysis_id>', methods=['GET'])
def get_lidar_analysis(analysis_id):
    """Get LIDAR analysis results"""
    try:
        analysis = LIDARAnalysis.query.get_or_404(analysis_id)
        
        return jsonify({
            'analysis_id': analysis.id,
            'scan_id': analysis.scan_id,
            'analysis_type': analysis.analysis_type,
            'risk_level': analysis.risk_level,
            'stability_score': analysis.stability_score,
            'timestamp': analysis.timestamp.isoformat(),
            'processing_time': analysis.processing_time,
            'results': json.loads(analysis.results_json)
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get analysis: {str(e)}'}), 500

@app.route('/api/lidar/scan/<int:scan_id>/analyses', methods=['GET'])
def get_scan_analyses(scan_id):
    """Get all analyses for a specific scan"""
    try:
        scan = LIDARScan.query.get_or_404(scan_id)
        analyses = LIDARAnalysis.query.filter_by(scan_id=scan_id).order_by(LIDARAnalysis.timestamp.desc()).all()
        
        analysis_list = []
        for analysis in analyses:
            analysis_list.append({
                'analysis_id': analysis.id,
                'analysis_type': analysis.analysis_type,
                'risk_level': analysis.risk_level,
                'stability_score': analysis.stability_score,
                'timestamp': analysis.timestamp.isoformat(),
                'processing_time': analysis.processing_time
            })
        
        return jsonify({
            'scan_id': scan_id,
            'scan_filename': scan.filename,
            'analyses': analysis_list
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get analyses: {str(e)}'}), 500

@app.route('/api/lidar/risk-map', methods=['GET'])
def get_lidar_risk_map():
    """Generate risk map from all recent LIDAR analyses"""
    try:
        # Get recent analyses
        recent_analyses = LIDARAnalysis.query.filter(
            LIDARAnalysis.timestamp >= datetime.utcnow() - timedelta(days=7)
        ).all()
        
        risk_zones = []
        for analysis in recent_analyses:
            scan = analysis.scan
            if scan.bounds_json:
                bounds = json.loads(scan.bounds_json)
                
                # Create risk zone from scan bounds
                center_x = (bounds['x'][0] + bounds['x'][1]) / 2
                center_y = (bounds['y'][0] + bounds['y'][1]) / 2
                
                # Convert to approximate lat/lng (this would need proper coordinate transformation in production)
                lat = center_y / 111000  # Very rough approximation
                lng = center_x / 111000
                
                risk_value = 1.0 - analysis.stability_score if analysis.stability_score else 0.5
                
                risk_zones.append({
                    'lat': lat,
                    'lng': lng,
                    'risk_value': risk_value,
                    'risk_level': analysis.risk_level,
                    'scan_id': scan.id,
                    'analysis_id': analysis.id,
                    'timestamp': analysis.timestamp.isoformat()
                })
        
        return jsonify({'risk_zones': risk_zones})
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate risk map: {str(e)}'}), 500

# ==================== END LIDAR API ENDPOINTS ====================

# ==================== LSTM TIME SERIES API ENDPOINTS ====================

# Initialize LSTM predictor globally
lstm_predictor = None
lstm_model_path = os.path.join(os.path.dirname(__file__), '..', 'ml_models', 'lstm_rockfall_model.h5')
lstm_scaler_path = os.path.join(os.path.dirname(__file__), '..', 'ml_models', 'lstm_scaler.pkl')

def initialize_lstm():
    """Initialize LSTM predictor if available"""
    global lstm_predictor
    if LSTM_AVAILABLE and lstm_predictor is None:
        try:
            config = LSTMConfig()
            lstm_predictor = RockfallLSTMPredictor(config)
            
            # Try to load existing model
            if os.path.exists(lstm_model_path):
                lstm_predictor.load_model(lstm_model_path, lstm_scaler_path)
                print("LSTM model loaded successfully")
            else:
                print("No pre-trained LSTM model found")
        except Exception as e:
            print(f"Failed to initialize LSTM predictor: {e}")

@app.route('/api/lstm/train', methods=['POST'])
def train_lstm():
    """Train LSTM model with uploaded CSV data"""
    if not LSTM_AVAILABLE:
        return jsonify({'error': 'LSTM functionality not available'}), 500
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        os.makedirs(os.path.dirname(lstm_model_path), exist_ok=True)
        csv_path = os.path.join(os.path.dirname(lstm_model_path), 'training_data.csv')
        file.save(csv_path)
        
        # Initialize predictor
        config = LSTMConfig()
        global lstm_predictor
        lstm_predictor = RockfallLSTMPredictor(config)
        
        # Train model
        history = lstm_predictor.train(csv_path, lstm_model_path)
        
        return jsonify({
            'status': 'success',
            'message': 'LSTM model trained successfully',
            'training_history': {
                'final_accuracy': float(history['accuracy'][-1]) if history['accuracy'] else 0,
                'final_loss': float(history['loss'][-1]) if history['loss'] else 0,
                'epochs_trained': len(history['accuracy']) if history['accuracy'] else 0
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

@app.route('/api/lstm/predict', methods=['POST'])
def lstm_predict():
    """Make LSTM prediction on sequence data"""
    if not LSTM_AVAILABLE:
        return jsonify({'error': 'LSTM functionality not available'}), 500
    
    global lstm_predictor
    if lstm_predictor is None or not lstm_predictor.is_trained:
        return jsonify({'error': 'LSTM model not trained'}), 400
    
    try:
        data = request.get_json()
        
        if 'sequence' not in data:
            return jsonify({'error': 'No sequence data provided'}), 400
        
        sequence = np.array(data['sequence'])
        
        # Validate sequence shape
        if sequence.shape != (lstm_predictor.config.sequence_length, lstm_predictor.config.num_features):
            return jsonify({
                'error': f'Invalid sequence shape. Expected ({lstm_predictor.config.sequence_length}, {lstm_predictor.config.num_features}), got {sequence.shape}'
            }), 400
        
        # Make prediction
        predicted_class, confidence = lstm_predictor.predict(sequence)
        risk_level = RockfallLSTMPredictor.risk_level_to_string(predicted_class)
        
        return jsonify({
            'status': 'success',
            'prediction': {
                'risk_level': risk_level,
                'risk_class': int(predicted_class),
                'confidence': float(confidence),
                'timestamp': datetime.utcnow().isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/lstm/predict-realtime', methods=['GET'])
def lstm_predict_realtime():
    """Make real-time LSTM prediction using recent sensor data"""
    if not LSTM_AVAILABLE:
        return jsonify({'error': 'LSTM functionality not available'}), 500
    
    global lstm_predictor
    if lstm_predictor is None or not lstm_predictor.is_trained:
        return jsonify({'error': 'LSTM model not trained'}), 400
    
    try:
        # Get recent sensor data for sequence
        recent_data = SensorData.query.filter(
            SensorData.timestamp >= datetime.utcnow() - timedelta(hours=1)
        ).order_by(SensorData.timestamp.desc()).limit(lstm_predictor.config.sequence_length * 6).all()
        
        if len(recent_data) < lstm_predictor.config.sequence_length:
            return jsonify({'error': 'Insufficient recent data for prediction'}), 400
        
        # Convert to sequence format
        sequence_data = []
        sensor_types = ['displacement', 'strain', 'pore_pressure', 'temperature', 'rainfall', 'seismic']
        
        for i in range(lstm_predictor.config.sequence_length):
            time_step = []
            for sensor_type in sensor_types:
                # Find most recent data for this sensor type at this time step
                sensor_value = 0.0
                for data_point in recent_data[i*6:(i+1)*6]:
                    if data_point.sensor_type == sensor_type:
                        sensor_value = data_point.value
                        break
                time_step.append(sensor_value)
            sequence_data.append(time_step)
        
        sequence = np.array(sequence_data)
        
        # Make prediction
        predicted_class, confidence = lstm_predictor.predict(sequence)
        risk_level = RockfallLSTMPredictor.risk_level_to_string(predicted_class)
        
        return jsonify({
            'status': 'success',
            'prediction': {
                'risk_level': risk_level,
                'risk_class': int(predicted_class),
                'confidence': float(confidence),
                'timestamp': datetime.utcnow().isoformat(),
                'data_points_used': len(recent_data)
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Real-time prediction failed: {str(e)}'}), 500

@app.route('/api/lstm/status', methods=['GET'])
def lstm_status():
    """Get LSTM model status and information"""
    global lstm_predictor
    
    status = {
        'lstm_available': LSTM_AVAILABLE,
        'model_loaded': lstm_predictor is not None,
        'model_trained': lstm_predictor.is_trained if lstm_predictor else False,
        'model_file_exists': os.path.exists(lstm_model_path),
        'scaler_file_exists': os.path.exists(lstm_scaler_path)
    }
    
    if lstm_predictor and lstm_predictor.is_trained:
        status['model_config'] = {
            'sequence_length': lstm_predictor.config.sequence_length,
            'num_features': lstm_predictor.config.num_features,
            'lstm_units': lstm_predictor.config.lstm_units
        }
        
        if lstm_predictor.model:
            status['model_summary'] = lstm_predictor.get_model_summary()
    
    return jsonify(status)

@app.route('/api/lstm/retrain', methods=['POST'])
def retrain_lstm():
    """Retrain LSTM model with new data"""
    if not LSTM_AVAILABLE:
        return jsonify({'error': 'LSTM functionality not available'}), 500
    
    try:
        # Use existing training data or upload new data
        csv_path = os.path.join(os.path.dirname(lstm_model_path), 'training_data.csv')
        
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                file.save(csv_path)
        
        if not os.path.exists(csv_path):
            return jsonify({'error': 'No training data available'}), 400
        
        # Initialize new predictor
        config = LSTMConfig()
        global lstm_predictor
        lstm_predictor = RockfallLSTMPredictor(config)
        
        # Train model
        history = lstm_predictor.train(csv_path, lstm_model_path)
        
        return jsonify({
            'status': 'success',
            'message': 'LSTM model retrained successfully',
            'training_history': {
                'final_accuracy': float(history['accuracy'][-1]) if history['accuracy'] else 0,
                'final_loss': float(history['loss'][-1]) if history['loss'] else 0,
                'epochs_trained': len(history['accuracy']) if history['accuracy'] else 0
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Retraining failed: {str(e)}'}), 500

# Auto-Training Endpoints
@app.route('/api/auto-train', methods=['POST'])
def auto_train_models():
    """Automatically train all models with uploaded data"""
    if not AUTO_TRAINING_AVAILABLE:
        return jsonify({'error': 'Auto-training not available'}), 500
    
    try:
        uploaded_file = None
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                # Save uploaded file temporarily
                filename = secure_filename(file.filename)
                upload_path = os.path.join('temp_uploads', filename)
                os.makedirs('temp_uploads', exist_ok=True)
                file.save(upload_path)
                uploaded_file = upload_path
        
        # Perform auto-training
        if uploaded_file:
            training_results = auto_trainer.auto_train_on_upload(uploaded_file)
            # Clean up temp file
            if os.path.exists(uploaded_file):
                os.remove(uploaded_file)
        else:
            # Train with generated data
            training_data = auto_trainer.generate_comprehensive_training_data()
            traditional_results = auto_trainer.train_traditional_models(training_data)
            lstm_results = auto_trainer.train_lstm_model(training_data)
            
            training_results = {
                'traditional_models': traditional_results,
                'lstm_model': lstm_results,
                'data_samples': len(training_data)
            }
        
        return jsonify({
            'status': 'success',
            'message': 'Auto-training completed successfully',
            'results': training_results
        })
        
    except Exception as e:
        return jsonify({'error': f'Auto-training failed: {str(e)}'}), 500

@app.route('/api/generate-training-data', methods=['POST'])
def generate_training_data():
    """Generate comprehensive training data"""
    if not AUTO_TRAINING_AVAILABLE:
        return jsonify({'error': 'Auto-training not available'}), 500
    
    try:
        # Get parameters
        n_samples = request.json.get('n_samples', 10000) if request.json else 10000
        
        # Generate data
        training_data = auto_trainer.generate_comprehensive_training_data(n_samples)
        
        # Return preview and statistics
        preview_data = training_data.head(100).to_dict('records')
        
        stats = {
            'total_samples': len(training_data),
            'risk_level_distribution': training_data['risk_level'].value_counts().to_dict(),
            'rockfall_occurrence_rate': float(training_data['rockfall_occurred'].mean()),
            'feature_columns': list(training_data.columns),
            'date_range': {
                'start': training_data['timestamp'].min().isoformat(),
                'end': training_data['timestamp'].max().isoformat()
            }
        }
        
        return jsonify({
            'status': 'success',
            'message': 'Training data generated successfully',
            'statistics': stats,
            'data_preview': preview_data
        })
        
    except Exception as e:
        return jsonify({'error': f'Data generation failed: {str(e)}'}), 500

@app.route('/api/model-status', methods=['GET'])
def get_model_status():
    """Get comprehensive status of all models"""
    try:
        status = {
            'lstm_available': LSTM_AVAILABLE,
            'auto_training_available': AUTO_TRAINING_AVAILABLE,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if AUTO_TRAINING_AVAILABLE:
            model_status = auto_trainer.get_model_status()
            status.update(model_status)
        
        if LSTM_AVAILABLE and lstm_predictor:
            status['lstm_model'] = {
                'loaded': lstm_predictor is not None,
                'trained': lstm_predictor.is_trained if lstm_predictor else False
            }
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': f'Status check failed: {str(e)}'}), 500

# ==================== NEW FEATURES API ENDPOINTS ====================

@app.route('/api/soil-rock/classify', methods=['POST'])
def classify_soil_rock():
    """Classify soil/rock type from uploaded image"""
    try:
        if not SOIL_CLASSIFICATION_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Soil/Rock classification service not available'
            }), 503
        
        # Get image data from request
        if 'image' not in request.files and 'image_data' not in request.json:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        # Handle file upload
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No file selected'
                }), 400
            
            # Save temporarily and classify
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                file.save(temp_file.name)
                result = soil_rock_classifier.predict(temp_file.name)
                os.unlink(temp_file.name)
        
        # Handle base64 image data
        elif 'image_data' in request.json:
            image_data = request.json['image_data']
            result = soil_rock_classifier.predict(image_data)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/soil-rock/info', methods=['GET'])
def get_soil_rock_info():
    """Get information about soil/rock classification model"""
    try:
        if not SOIL_CLASSIFICATION_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Soil/Rock classification service not available'
            }), 503
        
        info = soil_rock_classifier.get_model_info()
        return jsonify({
            'success': True,
            'info': info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/soil-rock/train', methods=['POST'])
def train_soil_rock_model():
    """Train soil/rock classification model with provided dataset"""
    try:
        if not SOIL_CLASSIFICATION_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Soil/Rock classification service not available'
            }), 503
        
        # Get dataset path from request
        data = request.get_json()
        dataset_path = data.get('dataset_path', 'd:/Datasets/Soil Test')
        
        if not os.path.exists(dataset_path):
            return jsonify({
                'success': False,
                'error': f'Dataset path not found: {dataset_path}'
            }), 400
        
        # Start training (this might take a while)
        history = soil_rock_classifier.train_model(dataset_path)
        
        return jsonify({
            'success': True,
            'message': 'Model training completed',
            'training_history': history.history if history else None
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/reports/email', methods=['POST'])
def send_email_report():
    """Send risk analysis report via email"""
    try:
        if not EMAIL_REPORTING_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Email reporting service not available'
            }), 503
        
        data = request.get_json()
        recipient_email = data.get('email')
        
        if not recipient_email:
            return jsonify({
                'success': False,
                'error': 'Email address is required'
            }), 400
        
        # Validate email configuration
        email_config = email_report_service.validate_email_config()
        if not email_config['valid']:
            return jsonify({
                'success': False,
                'error': f'Email configuration incomplete. Missing: {", ".join(email_config["missing_config"])}'
            }), 503
        
        # Get recent data for report
        recent_sensor_data = SensorData.query.filter(
            SensorData.timestamp >= datetime.utcnow() - timedelta(hours=24)
        ).all()
        
        recent_risk_assessments = RiskAssessment.query.filter(
            RiskAssessment.timestamp >= datetime.utcnow() - timedelta(hours=24)
        ).all()
        
        recent_alerts = Alert.query.filter(
            Alert.timestamp >= datetime.utcnow() - timedelta(hours=24)
        ).all()
        
        # Get current risk assessment
        current_risk = RiskAssessment.query.order_by(RiskAssessment.timestamp.desc()).first()
        
        # Prepare report data
        report_data = {
            'site_location': data.get('site_location', 'Mine Site Alpha'),
            'current_risk_level': current_risk.risk_level if current_risk else 'UNKNOWN',
            'current_probability': current_risk.probability if current_risk else 0,
            'active_sensors': len(set([s.sensor_id for s in recent_sensor_data])),
            'total_alerts': len(recent_alerts),
            'sensor_data': [
                {
                    'sensor_type': s.sensor_type,
                    'value': s.value,
                    'timestamp': s.timestamp.isoformat(),
                    'location_x': s.location_x,
                    'location_y': s.location_y
                }
                for s in recent_sensor_data
            ],
            'risk_assessments': [
                {
                    'risk_level': r.risk_level,
                    'probability': r.probability,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in recent_risk_assessments
            ],
            'recommendations': [
                'Continue monitoring sensor networks',
                'Maintain evacuation readiness',
                'Review safety protocols',
                'Check equipment status daily'
            ]
        }
        
        # Add custom recommendations based on risk level
        if current_risk:
            if current_risk.risk_level == 'HIGH':
                report_data['recommendations'].extend([
                    'Consider temporary evacuation of high-risk areas',
                    'Increase monitoring frequency to every 5 minutes'
                ])
            elif current_risk.risk_level == 'CRITICAL':
                report_data['recommendations'].extend([
                    'IMMEDIATE EVACUATION REQUIRED',
                    'Activate emergency response protocol',
                    'Contact emergency services'
                ])
        
        # Send email
        include_pdf = data.get('include_pdf', True)
        result = email_report_service.send_email_report(
            recipient_email, 
            report_data, 
            include_pdf=include_pdf
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/dashboard/enhanced-stats', methods=['GET'])
def get_enhanced_dashboard_stats():
    """Get enhanced dashboard statistics for the modern UI"""
    try:
        # Get recent data (last 24 hours)
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
        
        # High risk mines count
        high_risk_assessments = RiskAssessment.query.filter(
            RiskAssessment.timestamp >= recent_cutoff,
            RiskAssessment.risk_level.in_(['HIGH', 'CRITICAL'])
        ).count()
        
        # Total incidents (alerts)
        total_incidents = Alert.query.filter(
            Alert.timestamp >= recent_cutoff
        ).count()
        
        # Injuries (simulated based on alerts)
        injuries_6m = max(0, total_incidents - random.randint(100, 120))
        
        # Active mines (simulated)
        active_mines = 167
        
        # Risk distribution
        risk_counts = {
            'HIGH': RiskAssessment.query.filter(
                RiskAssessment.timestamp >= recent_cutoff,
                RiskAssessment.risk_level == 'HIGH'
            ).count(),
            'MEDIUM': RiskAssessment.query.filter(
                RiskAssessment.timestamp >= recent_cutoff,
                RiskAssessment.risk_level == 'MEDIUM'
            ).count(),
            'LOW': RiskAssessment.query.filter(
                RiskAssessment.timestamp >= recent_cutoff,
                RiskAssessment.risk_level == 'LOW'
            ).count()
        }
        
        total_assessments = sum(risk_counts.values()) or 1  # Avoid division by zero
        
        risk_distribution = {
            'high_risk': round((risk_counts['HIGH'] / total_assessments) * 100, 1),
            'medium_risk': round((risk_counts['MEDIUM'] / total_assessments) * 100, 1), 
            'low_risk': round((risk_counts['LOW'] / total_assessments) * 100, 1)
        }
        
        # India mining risk map data (simulated realistic locations)
        india_risk_map = [
            {'lat': 23.7644, 'lng': 86.4131, 'risk': 'HIGH', 'name': 'Jharia Coalfield'},
            {'lat': 22.9734, 'lng': 78.6569, 'risk': 'MEDIUM', 'name': 'Singrauli'},
            {'lat': 21.2787, 'lng': 81.8661, 'risk': 'MEDIUM', 'name': 'Korba'},
            {'lat': 12.9716, 'lng': 77.5946, 'risk': 'LOW', 'name': 'Kolar Gold Fields'},
            {'lat': 15.8281, 'lng': 74.4978, 'risk': 'HIGH', 'name': 'Goa Iron Ore'},
            {'lat': 22.4707, 'lng': 88.1682, 'risk': 'MEDIUM', 'name': 'Raniganj'}
        ]
        
        # Recent events
        recent_alerts = Alert.query.filter(
            Alert.timestamp >= recent_cutoff
        ).order_by(Alert.timestamp.desc()).limit(5).all()
        
        recent_events = []
        for alert in recent_alerts:
            severity_map = {
                'HIGH': 'Critical',
                'MEDIUM': 'High', 
                'LOW': 'Medium'
            }
            recent_events.append({
                'title': alert.message,
                'severity': severity_map.get(alert.severity, 'Medium'),
                'timestamp': alert.timestamp.isoformat()
            })
        
        # Weather triggers (simulated)
        weather_triggers = {
            'rainfall': {
                'value': random.randint(15, 35),
                'unit': 'mm/hr',
                'status': 'warning' if random.random() > 0.7 else 'normal'
            },
            'temperature_swing': {
                'value': random.randint(8, 18),
                'unit': '°C',
                'status': 'normal'
            },
            'wind': {
                'value': 'Normal',
                'status': 'normal'
            }
        }
        
        return jsonify({
            'success': True,
            'stats': {
                'high_risk_mines': high_risk_assessments,
                'total_incidents': total_incidents,
                'injuries_6m': injuries_6m,
                'active_mines': active_mines
            },
            'risk_distribution': risk_distribution,
            'india_risk_map': india_risk_map,
            'recent_events': recent_events,
            'weather_triggers': weather_triggers,
            'last_updated': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/mine-site/details', methods=['GET'])
def get_mine_site_details():
    """Get detailed mine site information for 3D visualization"""
    try:
        site_id = request.args.get('site_id', 'jharia-section-a')
        
        # Simulated mine site data
        mine_sites = {
            'jharia-section-a': {
                'name': 'Jharia Coalfield - Section A',
                'location': 'Jharkhand, India',
                'coordinates': {'lat': 23.7644, 'lng': 86.4131},
                'status': 'Active Mine',
                'last_inspection': '3 days ago',
                'sensors': [
                    {
                        'id': 'TLT-001',
                        'type': 'Tiltmeter',
                        'sector': 'Sector 7-North',
                        'description': 'Monitoring slope stability on north wall',
                        'status': 'active',
                        'value': 0.8,
                        'unit': 'degrees',
                        'threshold': 1.0
                    },
                    {
                        'id': 'PZ-001', 
                        'type': 'Piezometer',
                        'sector': 'Sector 5-Ground',
                        'description': 'Groundwater pressure monitoring',
                        'status': 'warning',
                        'value': 45,
                        'unit': 'kPa',
                        'threshold': 50
                    },
                    {
                        'id': 'VIB-001',
                        'type': 'Vibration',
                        'sector': 'Sector 3-Blast',
                        'description': 'Blast vibration monitoring',
                        'status': 'active',
                        'value': 0.8,
                        'unit': 'mm/s',
                        'threshold': 1.0
                    },
                    {
                        'id': 'CRACK-001',
                        'type': 'Crackmeter',
                        'sector': 'Sector 7-Critical',
                        'description': 'Rock crack width monitoring',
                        'status': 'critical',
                        'value': 2.5,
                        'unit': 'mm',
                        'threshold': 2.0
                    }
                ],
                'risk_zones': [
                    {
                        'name': 'ZONE-A',
                        'risk_level': 'MEDIUM',
                        'type': 'Vibration',
                        'coordinates': [23.7650, 86.4135],
                        'description': 'Sector 3-Blast'
                    },
                    {
                        'name': 'ZONE-B', 
                        'risk_level': 'HIGH',
                        'type': 'Crackmeter',
                        'coordinates': [23.7648, 86.4128],
                        'description': 'ZONE-C - HIGH RISK'
                    },
                    {
                        'name': 'ZONE-C',
                        'risk_level': 'HIGH',
                        'type': 'Crackmeter', 
                        'coordinates': [23.7642, 86.4132],
                        'description': 'ZONE-C - HIGH RISK'
                    }
                ],
                'current_readings': {
                    'strain': {'value': 78, 'unit': 'μe', 'threshold': 75, 'status': 'warning'},
                    'temperature': {'value': 32, 'unit': '°C', 'threshold': 40, 'status': 'normal'},
                    'rainfall': {'value': 125, 'unit': 'mm', 'threshold': 100, 'status': 'warning'},
                    'pore_pressure': {'value': 45, 'unit': 'kPa', 'threshold': 50, 'status': 'warning'},
                    'slope_angle': {'value': 67, 'unit': '°', 'threshold': 60, 'status': 'critical'},
                    'vibration': {'value': 0.8, 'unit': 'mm/s', 'threshold': 1.0, 'status': 'normal'}
                }
            }
        }
        
        site_data = mine_sites.get(site_id, mine_sites['jharia-section-a'])
        
        return jsonify({
            'success': True,
            'site_data': site_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ==================== END NEW FEATURES API ENDPOINTS ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Comprehensive health check endpoint"""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'database': 'connected',
            'services': {
                'lstm': LSTM_AVAILABLE,
                'auto_training': AUTO_TRAINING_AVAILABLE,
                'lidar_processing': 'lidar_data_handler' in globals(),
                'alert_service': True
            }
        }
        
        # Test database connection
        try:
            db.session.execute('SELECT 1')
            health_status['database'] = 'Connected'
        except Exception as e:
            health_status['database'] = f'Error: {str(e)}'
            health_status['status'] = 'degraded'
        
        # Check model files
        if AUTO_TRAINING_AVAILABLE:
            model_status = auto_trainer.get_model_status()
            health_status['ml_models'] = 'Loaded' if any([
                model_status['traditional_models']['classifier_available'],
                model_status['traditional_models']['regressor_available'],
                model_status['lstm_model']['model_loaded']
            ]) else 'Not Loaded'
        else:
            health_status['ml_models'] = 'Auto-training not available'
        
        return jsonify(health_status)
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500
        
    except Exception as e:
        return jsonify({'error': f'Retraining failed: {str(e)}'}), 500

# ==================== END LSTM API ENDPOINTS ====================

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Initialize LSTM if available
        initialize_lstm()
    app.run(debug=True, host='0.0.0.0', port=5000)