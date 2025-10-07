import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json

class PredictiveFeatureExtractor:
    def __init__(self, enhanced_json):
        self.enhanced_json = enhanced_json
        self.features = {}
        self.global_patterns = {}
    # all features for predictive monitoring    
    def extract_all_features(self):
        # Extract global patterns first
        self._extract_global_patterns()
        
        # Extract features for each entity
        for entity_id, entity_data in self.enhanced_json['entities'].items():
            self.features[entity_id] = self._extract_entity_features(entity_id, entity_data)
        
        print(f"Extracted features for {len(self.features)} entities")
        return self.features, self.global_patterns
    # extract campus wide pattern
    def _extract_global_patterns(self):        
        # Department location preferences
        department_locations = defaultdict(list)
        hour_activity = defaultdict(int)
        location_popularity = defaultdict(int)
        
        for entity_id, entity_data in self.enhanced_json['entities'].items():
            department = entity_data['profile_info']['department']
            behavioral = entity_data.get('behavioral_patterns', {})
            temporal = entity_data.get('temporal_analysis', {})
            
            # Department patterns
            if 'unique_locations' in behavioral:
                department_locations[department].extend(behavioral['unique_locations'])
            
            # Hourly activity patterns
            if 'hourly_activity_distribution' in temporal:
                for hour, count in temporal['hourly_activity_distribution'].items():
                    hour_activity[hour] += count
            
            # Location popularity
            if 'location_frequency' in behavioral:
                for location, freq in behavioral['location_frequency'].items():
                    location_popularity[location] += freq
        
        # Calculate most common patterns
        self.global_patterns = {
            'department_location_preferences': {
                dept: Counter(locs).most_common(5) for dept, locs in department_locations.items()
            },
            'campus_peak_hours': dict(sorted(hour_activity.items(), key=lambda x: x[1], reverse=True)[:5]),
            'popular_locations': dict(sorted(location_popularity.items(), key=lambda x: x[1], reverse=True)[:10]),
            'location_categories': self._categorize_locations(location_popularity)
        }
        
        print(f"Found {len(self.global_patterns['department_location_preferences'])} department patterns")
        print(f"Peak hours: {list(self.global_patterns['campus_peak_hours'].keys())}")
    
    def _extract_entity_features(self, entity_id, entity_data):
        profile = entity_data['profile_info']
        behavioral = entity_data.get('behavioral_patterns', {})
        temporal = entity_data.get('temporal_analysis', {})
        location_analysis = entity_data.get('location_analysis', {})
        ml_features = entity_data.get('ml_features', {})
        timeline = entity_data.get('activity_timeline', [])
        
        features = {
            # Basic identity features
            'entity_id': entity_id,
            'department': profile.get('department', 'Unknown'),
            'role': profile.get('role', 'Unknown'),
            
            # Temporal patterns
            'temporal_features': self._extract_temporal_features(temporal, timeline),
            
            # Location preferences 
            'location_features': self._extract_location_features(behavioral, location_analysis),
            
            # Behavioral sequences
            'sequence_features': self._extract_sequence_features(behavioral, timeline),
            
            # Activity patterns
            'activity_features': self._extract_activity_features(ml_features, timeline),
            
            # Contextual features
            'context_features': self._extract_context_features(entity_data),
            
            # Predictive signals
            'predictive_signals': self._extract_predictive_signals(entity_data)
        }
        
        return features
    #extract temporal features for prediction
    def _extract_temporal_features(self, temporal, timeline):
        features = {}
        
        # From temporal_analysis
        if 'hourly_activity_distribution' in temporal:
            hourly = temporal['hourly_activity_distribution']
            features.update({
                'peak_activity_hours': list(hourly.keys()),
                'most_active_hour': max(hourly.items(), key=lambda x: x[1])[0] if hourly else None,
                'activity_by_time_period': self._calculate_time_period_activity(hourly),
                'is_morning_person': sum(hourly.get(h, 0) for h in [6,7,8,9]) > sum(hourly.get(h, 0) for h in [21,22,23,0]),
                'is_night_owl': sum(hourly.get(h, 0) for h in [21,22,23,0,1,2]) > sum(hourly.get(h, 0) for h in [6,7,8,9,10,11])
            })
        
        if 'weekday_vs_weekend_ratio' in temporal:
            features['weekend_activity_ratio'] = temporal['weekday_vs_weekend_ratio']
        
        if 'most_active_day' in temporal:
            features['preferred_weekday'] = temporal['most_active_day']
        
        # Enhanced temporal features from timeline
        if timeline:
            timestamps = [pd.to_datetime(item['timestamp']) for item in timeline if item.get('timestamp')]
            if timestamps:
                features.update({
                    'days_since_first_activity': (datetime.now() - min(timestamps)).days,
                    'days_since_last_activity': (datetime.now() - max(timestamps)).days,
                    'activity_regularity': self._calculate_regularity_score(timestamps)
                })
        
        return features
    # extract location features for prediction
    def _extract_location_features(self, behavioral, location_analysis):
        features = {}
        
        # From behavioral_patterns
        if 'location_frequency' in behavioral:
            loc_freq = behavioral['location_frequency']
            features.update({
                'frequent_locations': list(loc_freq.keys()),
                'location_diversity': len(loc_freq),
                'most_visited_location': max(loc_freq.items(), key=lambda x: x[1])[0] if loc_freq else None,
                'location_entropy': behavioral.get('location_entropy', 0),
                'visit_frequency': behavioral.get('visit_frequency', 0)
            })
        
        # From location_analysis  
        if 'location_preferences_by_time' in location_analysis:
            time_prefs = location_analysis['location_preferences_by_time']
            features['time_based_preferences'] = time_prefs
            
            # Extract strongest time-location associations
            strong_associations = []
            for time_period, locations in time_prefs.items():
                for location, confidence in locations.items():
                    if confidence > 0.3:  # Threshold for strong association
                        strong_associations.append(f"{time_period}_{location}")
            
            features['strong_time_location_associations'] = strong_associations
        
        if 'most_visited_location' in location_analysis:
            features['primary_location'] = location_analysis['most_visited_location']
        
        return features
    # movement sequence patterns
    def _extract_sequence_features(self, behavioral, timeline):
        features = {}
        
        # From behavioral_patterns
        if 'common_transitions' in behavioral:
            transitions = behavioral['common_transitions']
            features.update({
                'common_movements': list(transitions.keys()),
                'most_common_transition': max(transitions.items(), key=lambda x: x[1])[0] if transitions else None,
                'transition_variety': len(transitions),
                'sequence_consistency': behavioral.get('activity_consistency', 0)
            })
        
        if 'location_sequence' in behavioral:
            sequence = behavioral['location_sequence']
            features.update({
                'full_location_sequence': sequence,
                'sequence_length': len(sequence),
                'recent_sequence': sequence[-3:] if len(sequence) >= 3 else sequence
            })
        
        # Extract recent activity context
        if timeline:
            recent_activities = sorted(timeline, key=lambda x: x['timestamp'], reverse=True)[:5]
            features['recent_activities'] = [
                {
                    'location': act.get('location'),
                    'activity_type': act.get('activity_type'),
                    'hours_ago': self._hours_from_now(act.get('timestamp')),
                    'time_period': self._get_time_period_from_hour(pd.to_datetime(act.get('timestamp')).hour if act.get('timestamp') else None)
                }
                for act in recent_activities
            ]
        
        return features
    # general activity patterns
    def _extract_activity_features(self, ml_features, timeline):
        features = {}
        
        # From ml_features
        features.update({
            'total_activities': ml_features.get('total_activities', 0),
            'activity_variety': ml_features.get('activity_variety', 0),
            'data_sources_used': ml_features.get('data_sources_used', []),
            'location_consistency': ml_features.get('location_consistency', 0)
        })
        
        # Calculate activity 
        if timeline:
            timestamps = [pd.to_datetime(item['timestamp']) for item in timeline if item.get('timestamp')]
            if timestamps and len(timestamps) > 1:
                time_span = (max(timestamps) - min(timestamps)).total_seconds() / 3600  # hours
                features['activity_density'] = len(timestamps) / max(time_span, 1)  # activities per hour
        
        return features
    # context features 
    def _extract_context_features(self, entity_data):
        profile = entity_data['profile_info']
        department = profile.get('department', 'Unknown')
        
        # Department-based context
        dept_patterns = self.global_patterns['department_location_preferences'].get(department, [])
        campus_peak_hours = self.global_patterns['campus_peak_hours']
        
        return {
            'department_common_locations': [loc for loc, freq in dept_patterns],
            'campus_peak_hours': list(campus_peak_hours.keys()),
            'current_time_context': self._get_current_time_context(),
            'location_categories': self.global_patterns['location_categories']
        }
    
    def _extract_predictive_signals(self, entity_data):
        behavioral = entity_data.get('behavioral_patterns', {})
        temporal = entity_data.get('temporal_analysis', {})
        location_analysis = entity_data.get('location_analysis', {})
        
        signals = {}
        
        # Time-based prediction signals
        if 'hourly_activity_distribution' in temporal:
            hourly = temporal['hourly_activity_distribution']
            current_hour = datetime.now().hour
            
            # Predictability based on historical patterns
            signals['time_based_predictability'] = hourly.get(current_hour, 0) / max(sum(hourly.values()), 1)
            
            # Suggested locations for current time
            time_prefs = location_analysis.get('location_preferences_by_time', {})
            current_time_period = self._get_current_time_period()
            signals['suggested_locations_current_time'] = time_prefs.get(current_time_period, {})
        
        # Sequence-based signals
        if 'common_transitions' in behavioral:
            transitions = behavioral['common_transitions']
            if transitions:
                signals['most_likely_next_movement'] = max(transitions.items(), key=lambda x: x[1])[0]
        
        # Department pattern signals
        department = entity_data['profile_info'].get('department')
        dept_patterns = self.global_patterns['department_location_preferences'].get(department, [])
        if dept_patterns:
            signals['department_suggested_locations'] = [loc for loc, freq in dept_patterns[:3]]
        
        return signals
    
    # calculation
    
    def _calculate_time_period_activity(self, hourly_distribution):
        """Calculate activity by time periods"""
        periods = {
            'morning': [6, 7, 8, 9, 10, 11],
            'afternoon': [12, 13, 14, 15, 16, 17], 
            'evening': [18, 19, 20, 21],
            'night': [22, 23, 0, 1, 2, 3, 4, 5]
        }
        
        period_activity = {}
        for period, hours in periods.items():
            period_activity[period] = sum(hourly_distribution.get(h, 0) for h in hours)
        
        return period_activity
    
    def _calculate_regularity_score(self, timestamps):
        """Calculate how regular the activity patterns are"""
        if len(timestamps) < 2:
            return 0
        
        # Calculate time differences between consecutive activities
        timestamps.sort()
        differences = []
        for i in range(1, len(timestamps)):
            diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600  
            differences.append(diff)
        
        # Lower std dev = more regular
        if differences:
            return 1 / (1 + np.std(differences))  # Convert to 0-1 score
        return 0
    # categorized locations 
    def _categorize_locations(self, location_popularity):
        categories = {
            'academic': ['LIB', 'LAB', 'SEM', 'ROOM', 'AP_ENG', 'AP_LAB'],
            'residential': ['HOSTEL', 'GATE'],
            'recreational': ['AUDITORIUM', 'AP_AUD'],
            'other': []
        }
        
        categorized = defaultdict(list)
        for location in location_popularity.keys():
            matched = False
            for category, keywords in categories.items():
                if any(keyword in location for keyword in keywords):
                    categorized[category].append(location)
                    matched = True
                    break
            if not matched:
                categorized['other'].append(location)
        
        return dict(categorized)
    
    def _hours_from_now(self, timestamp_str):
        """Calculate hours from now for a timestamp"""
        if not timestamp_str:
            return None
        try:
            timestamp = pd.to_datetime(timestamp_str)
            return (datetime.now() - timestamp).total_seconds() / 3600
        except:
            return None
    
    def _get_time_period_from_hour(self, hour):
        """Convert hour to time period"""
        if hour is None:
            return None
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    
    def _get_current_time_period(self):
        """Get current time period"""
        return self._get_time_period_from_hour(datetime.now().hour)
    
    def _get_current_time_context(self):
        """Get current time context for prediction"""
        current_time = datetime.now()
        return {
            'current_hour': current_time.hour,
            'current_time_period': self._get_current_time_period(),
            'is_weekend': current_time.weekday() >= 5,
            'day_of_week': current_time.weekday()
        }


# extract predictive features on json
def extract_features_from_json(enhanced_json):
    
    extractor = PredictiveFeatureExtractor(enhanced_json)
    features, global_patterns = extractor.extract_all_features()
    
    
    # Show sample features
    if features:
        first_entity = list(features.keys())[0]
        sample_features = features[first_entity]
        print(f"Temporal features: {len(sample_features['temporal_features'])}")
        print(f"Location features: {len(sample_features['location_features'])}")
        print(f"Sequence features: {len(sample_features['sequence_features'])}")
        print(f"Predictive signals: {len(sample_features['predictive_signals'])}")
    
    return features, global_patterns

# Example usage with your JSON
if __name__ == "__main__":
    # Load your enhanced JSON
    with open('Entity_resolution_map_code_file', 'r') as f:
       enhanced_json= json.load(f)
       
    
    features, global_patterns = extract_features_from_json(enhanced_json)
    
    # Save features for ML training
    output_data = {
        'features': features,
        'global_patterns': global_patterns,
        'extraction_timestamp': datetime.now().isoformat()
    }
    
    with open('predictive_features.json', 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

