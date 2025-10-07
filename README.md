A comprehensive entity resolution and security monitoring system that unifies campus data sources to provide proactive threat detection and explainable AI predictions.


security_monitoring_system/
->  Core Application Files
-> security_dashboard.py ( # Main Streamlit UI)

-> production_predictor.py (# ML prediction backend)
-> pipeline.py (# ML training pipeline)
-> EntityResolver.py ( # Entity resolution pipeline)



  Machine Learning
   -> trained_model.joblib           # trained ML model
   -> predictive_features.json       # Entity features for ML
   ->predictive_features_code_file.py # Feature generation code


 Entity Resolution
   -> Entity_resolution_map.json     # Entity mapping data
   . Entity_resolution_map_code_file.py # Resolution logic



 Raw Data Sources
   . campus card_swipes.csv         # Card swipe access logs
   .wifi_associations_logs.csv     # WiFi connection data
   . cctv_frames.csv               # CCTV face recognition data
   . face_embeddings.csv           # Facial feature vectors
   . library_checkouts.csv         #lab record records
   . lab_bookings.csv              # Laboratory reservations
   . free_text_notes.csv           # Helpdesk tickets & RSVPs
   .student or staff profiles.csv # Entity master data



├Documentation
  ├── README.md                      # This file
  └── report.pdf                     # Technical report

 

RAW DATA SOURCES->
    
DATA INGESTION LAYER
• CSV File Parsing
• Data Validation  
• Timestamp Normalization->
    
ENTITY RESOLUTION ENGINE
• Direct Matching (student_id, card_id, face_id)
• Fuzzy Matching (Name, Email variants)
• Cross-Source Linking (Temporal, Spatial)
    
MULTI-MODAL FUSION LAYER
• Temporal Alignment (5-min windows)
• Confidence-weighted Data Fusion
• Activity Timeline Generation
    
MACHINE LEARNING PIPELINE  
• Feature Engineering (Temporal, Spatial, Behavioral, Sequential)
• Model Training (XGBoost, Random Forest, Ensemble)
• Prediction & Evidence Generation

    
SECURITY DASHBOARD
• Real-time Entity Monitoring
• Predictive Location Insights
• 12-hour Inactivity Alerts

