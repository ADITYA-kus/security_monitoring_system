Campus Security Monitoring System 

A comprehensive entity resolution and security monitoring system that unifies campus data sources to provide proactive threat detection and explainable AI predictions.

#Features

Multi-Source Entity Resolution - Resolve identities across swipe logs, WiFi, CCTV, and text data
Real-time Security Monitoring - 12-hour inactivity alerts with configurable thresholds
AI-Powered Predictions - ML-based location prediction with evidence chains
Explainable AI - Transparent reasoning for all predictions
Privacy-Aware Design*- Data anonymization and access controls
Interactive Dashboard - Streamlit-based user interface

security_monitoring_system/
├── 🎯 Core Application Files
│   ├── security_dashboard.py          # Main Streamlit UI
│   ├── production_predictor.py        # ML prediction backend
│   ├── pipeline.py                    # ML training pipeline
│   └── EntityResolver.py              # Entity resolution engine
│
├── 🤖 Machine Learning
│   ├── trained_model.joblib           # Pre-trained ML model
│   ├── predictive_features.json       # Entity features for ML
│   └── predictive_features_code_file.py # Feature generation code
│
├── 🔗 Entity Resolution
│   ├── Entity_resolution_map.json     # Entity mapping data
│   └── Entity_resolution_map_code_file.py # Resolution logic
│
├── 📊 Raw Data Sources
│   ├── campus card_swipes.csv         # Card swipe access logs
│   ├── wifi_associations_logs.csv     # WiFi connection data
│   ├── cctv_frames.csv               # CCTV face recognition data
│   ├── face_embeddings.csv           # Facial feature vectors
│   ├── library_checkouts.csv         # Book borrowing records
│   ├── lab_bookings.csv              # Laboratory reservations
│   ├── free_text_notes.csv           # Helpdesk tickets & RSVPs
│   └── student or staff profiles.csv # Entity master data
│
├── 📄 Documentation
│   ├── README.md                      # This file
│   └── report.pdf                     # Technical report
│
└── ⚙️ Configuration
    └── requirements.txt               # Python dependencies

RAW DATA SOURCES
    ↓
DATA INGESTION LAYER
• CSV File Parsing
• Data Validation  
• Timestamp Normalization
    ↓
ENTITY RESOLUTION ENGINE
• Direct Matching (student_id, card_id, face_id)
• Fuzzy Matching (Name, Email variants)
• Cross-Source Linking (Temporal, Spatial)
    ↓
MULTI-MODAL FUSION LAYER
• Temporal Alignment (5-min windows)
• Confidence-weighted Data Fusion
• Activity Timeline Generation
    ↓
MACHINE LEARNING PIPELINE  
• Feature Engineering (Temporal, Spatial, Behavioral, Sequential)
• Model Training (XGBoost, Random Forest, Ensemble)
• Prediction & Evidence Generation
    ↓
SECURITY DASHBOARD
• Real-time Entity Monitoring
• Predictive Location Insights
• 12-hour Inactivity Alerts
