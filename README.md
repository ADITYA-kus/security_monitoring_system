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
├──  Core Application Files
│   ├── security_dashboard.py          # Main Streamlit UI
│   ├── production_predictor.py        # ML prediction backend
│   ├── pipeline.py                    # ML training pipeline
│   └── EntityResolver.py              # Entity resolution pipeline
│
├──  Machine Learning
│   ├── trained_model.joblib           # Pre-trained ML model
│   ├── predictive_features.json       # Entity features for ML
│   └── predictive_features_code_file.py # Feature generation code
│
├──  Entity Resolution
│   ├── Entity_resolution_map.json     # Entity mapping data
│   └── Entity_resolution_map_code_file.py # Resolution logic
│
├──  Documentation
│   ├── README.md                      # This file
│   └── report.pdf                     # Technical report
└── 

RAW DATA SOURCES
    ↓
┌─────────────────────────────────────────────────┐
│              DATA INGESTION LAYER               │
│  • CSV File Parsing                            │
│  • Data Validation                             │
│  • Timestamp Normalization                     │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│           ENTITY RESOLUTION ENGINE              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │ Direct      │ │ Fuzzy       │ │ Cross-      │ │
│  │ Matching    │ │ Matching    │ │ Source      │ │
│  │ • student_id│ │ • Name      │ │ Linking     │ │
│  │ • card_id   │ │ • Email     │ │ • Temporal  │ │
│  │ • face_id   │ │ variants    │ │ • Spatial   │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│           MULTI-MODAL FUSION LAYER              │
│  • Temporal Alignment (5-min windows)          │
│  • Confidence-weighted Data Fusion             │
│  • Activity Timeline Generation                │
│  • Gap Detection & Interpolation               │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│          MACHINE LEARNING PIPELINE              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │ Feature     │ │ Model       │ │ Prediction  │ │
│  │ Engineering │ │ Training    │ │ & Evidence  │ │
│  │ • Temporal  │ │ • XGBoost   │ │ • Location  │ │
│  │ • Spatial   │ │ • Random    │ │ • Confidence│ │
│  │ • Behavioral│ │ Forest      │ │ • Reasoning │ │
│  │ • Sequential│ │ • Ensemble  │ │ • Alerts    │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│            SECURITY DASHBOARD                   │
│  • Real-time Entity Monitoring                 │
│  • Predictive Location Insights                │
│  • 12-hour Inactivity Alerts                   │
│  • Explainable AI Evidence                     │
│  • Multi-source Activity Timeline              │
└─────────────────────────────────────────────────┘
