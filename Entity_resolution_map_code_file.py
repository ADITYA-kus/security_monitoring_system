import json
from datetime import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
from EntityResolver import CompleteEntityResolver

#load raw data
def load_all_datasets():
    datasets = {}
    
    try:
        datasets['profile'] = pd.read_csv('RAW_Data_folder\student or staff profiles.csv')
        datasets['wifi_logs'] = pd.read_csv('RAW_Data_folder\wifi_associations_logs.csv')
        datasets['campus_swipes'] = pd.read_csv('RAW_Data_folder\campus card_swipes.csv')
        datasets['library_check'] = pd.read_csv('RAW_Data_folder\library_checkouts.csv')
        datasets['lab_bookings'] = pd.read_csv('RAW_Data_folder\lab_bookings.csv')
        datasets['text_notes'] = pd.read_csv('RAW_Data_folder/free_text_notes (helpdesk or RSVPs).csv')
        datasets['face_vector'] = pd.read_csv('RAW_Data_folder/face_embeddings.csv')
        datasets['cctv_frame'] = pd.read_csv('RAW_Data_folder\cctv_frames.csv')
        
        print("all datasets load successfully!")
    except Exception as e:
        print(f" Error loading data: {e}")
        return None
    
    return datasets

# entity mapping using all identifier from profile
class CompleteFixedEntityResolver(CompleteEntityResolver):
    def _build_complete_entity_maps(self):        
        if 'profile' not in self.datasets:
            raise ValueError("Profile detaset not found")
        
        total_profiles = len(self.datasets['profile'])
        print(f"Processing {total_profiles} profiles")
        
        # Show identifiers we are working with
        sample_profile = self.datasets['profile'].iloc[0]
        print(f"   Sample profile identifiers:")
        for field in ['entity_id', 'student_id', 'staff_id', 'card_id', 'device_hash', 'face_id']:
            if field in sample_profile:
                print(f"{field}: {sample_profile[field]}")
        
        for idx, person in self.datasets['profile'].iterrows():
            entity_id = person['entity_id']
            
            # Extract identifiers
            identifiers = []
            id_fields = ['entity_id', 'student_id', 'staff_id', 'card_id', 'device_hash', 'face_id', 'email']
            
            for field in id_fields:
                if field in person and pd.notna(person[field]):
                    identifier_value = str(person[field])
                    identifiers.append(identifier_value)
                    
                    # Map thease identifier to the entity
                    self.id_to_entity[identifier_value] = entity_id
            
            # Storeing complete entity info
            self.entity_registry[entity_id] = {
                'name': person.get('name', 'Unknown'),
                'role': person.get('role', 'Unknown'),
                'email': person.get('email',''),
                'department': person.get('department',''),
                'all_identifiers': identifiers,
                'source': 'profile',
                'resolution_method': 'direct_mapping'
            }
        
    #linking with multiple matching
    def _link_all_data_sources(self):        
        linking_config = [
            ('wifi_logs', 'device_hash', 'wifi_logs'),
            ('campus_swipes', 'card_id', 'campus_swipes'),
            ('library_check', 'entity_id', 'library_checkouts'),
            ('lab_bookings', 'entity_id', 'lab_bookings'),
            ('text_notes', 'entity_id', 'text_notes'),
            ('face_vector', 'face_id', 'face_vectors'),
            ('cctv_frame', 'face_id', 'cctv_frames')
        ]
        
        total_linked = 0
        for dataset_name, id_field, activity_type in linking_config:
            if dataset_name in self.datasets:
                df = self.datasets[dataset_name]
                print(f"\n Processing {dataset_name}")
                
                # Check id field exists or not
                if id_field not in df.columns:
                    print(f"SKIPG: {id_field} column not found")
                    continue
                
                linked_count = 0
                sample_linked = []
                
                for _, record in df.iterrows():
                    identifier = record.get(id_field)
                    entity_id = self._enhanced_find_entity(identifier, id_field)
                    
                    if entity_id:
                        self.entity_activities[entity_id][activity_type].append({
                            'record': record.to_dict(),
                            'source': dataset_name,
                            'confidence': 1.0,
                            'provenance': f"direct_{id_field}_match",
                            'timestamp': self._extract_timestamp(record, activity_type)
                        })
                        linked_count += 1
                        if len(sample_linked) < 3:
                            sample_linked.append(f"{identifier} → {entity_id}")
                
                total_linked += linked_count
                print(f"Linked {linked_count}/{len(df)} records")
                if sample_linked:
                    print(f"Sample matches: {sample_linked}")
        
        print(f"\nTOTAL: {total_linked} activities linked")
    # entity finding with multiple strategies::
    def _enhanced_find_entity(self, identifier, id_field):
        if pd.isna(identifier):
            return None
        
        identifier_str = str(identifier)
        
        # 1: Direct match
        if identifier_str in self.id_to_entity:
            return self.id_to_entity[identifier_str]
        
        # 2: Case-insensitive match
        for key in self.id_to_entity.keys():
            if key.lower() == identifier_str.lower():
                return self.id_to_entity[key]
        
        # 3: For face_id try to match filename patterns (remove .jpg)
        if id_field == 'face_id' and '.jpg' in identifier_str:
            base_name = identifier_str.replace('.jpg', '')
            if base_name in self.id_to_entity:
                return self.id_to_entity[base_name]
        
        #  4: profile may be use ddifferent entty_id format than other datsets try to find any identifier that contains this value
        if id_field == 'entity_id':
            for key in self.id_to_entity.keys():
                if identifier_str in key or key in identifier_str:
                    return self.id_to_entity[key]
        
        return None

# generate final json for patterns analysis
class ImprovedEntityResolver(CompleteFixedEntityResolver):
    def generate_enhanced_json_output(self):
        
        enhanced_output = {
            'entities': self._generate_enhanced_entities(),
            'patterns_ready': True,
        }
        
        return enhanced_output
    
    # making entity data with pattern ready structure
    def _generate_enhanced_entities(self):
        entities = {}
        
        for entity_id in self.entity_registry.keys():
            entities[entity_id] = {
                'profile_info': self.entity_registry[entity_id],
                'activity_timeline': self._generate_activity_timeline(entity_id),
                'behavioral_patterns': self._extract_behavioral_patterns(entity_id),
                'location_analysis': self._analyze_location_patterns(entity_id),
                'temporal_analysis': self._analyze_temporal_patterns(entity_id),
                'evidence_chains': self._generate_evidence_chains(entity_id),
                'ml_features': self._extract_ml_features(entity_id)
            }
        
        return entities
    # activity timeline
    def _generate_activity_timeline(self, entity_id):
        timeline = []
        
        for activity_type, activities in self.entity_activities[entity_id].items():
            for activity in activities:
                timestamp = activity.get('timestamp')
                if timestamp:
                    timeline.append({
                        'timestamp': timestamp.isoformat(),
                        'activity_type': activity_type,
                        'location': self._extract_location(activity['record'], activity_type),
                        'source': activity['source'],
                        'confidence': activity['confidence'],
                        'provenance': activity['provenance'],
                        'details': self._clean_activity_details(activity['record'], activity_type)
                    })
        
        # Sort by timestamp
        timeline.sort(key=lambda x: x['timestamp'])
        return timeline
    # extract behavioral pattern
    def _extract_behavioral_patterns(self, entity_id):
        activities = self.entity_activities[entity_id]
        
        # Extract location sequences
        locations = []
        location_times = []
        
        for activity_type, activity_list in activities.items():
            for activity in activity_list:
                timestamp = activity.get('timestamp')
                location = self._extract_location(activity['record'], activity_type)
                
                if timestamp and location:
                    locations.append(location)
                    location_times.append({
                        'location': location,
                        'timestamp': timestamp,
                        'hour': timestamp.hour,
                        'day_of_week': timestamp.weekday(),
                        'is_weekend': timestamp.weekday() >= 5
                    })
        
        # Calculate patterns
        if location_times:
            location_times.sort(key=lambda x: x['timestamp'])
            location_sequence = [lt['location'] for lt in location_times]
            
            # Time-based patterns
            hourly_distribution = defaultdict(int)
            for lt in location_times:
                hourly_distribution[lt['hour']] += 1
            
            # Location frequency
            location_frequency = defaultdict(int)
            for location in locations:
                location_frequency[location] += 1
            
            # Movement patterns (transitions)
            transitions = []
            for i in range(1, len(location_sequence)):
                transitions.append(f"{location_sequence[i-1]}→{location_sequence[i]}")
            
            transition_frequency = defaultdict(int)
            for transition in transitions:
                transition_frequency[transition] += 1
            
            return {
                'location_sequence': location_sequence,
                'unique_locations': list(set(locations)),
                'location_frequency': dict(location_frequency),
                'hourly_distribution': dict(hourly_distribution),
                'common_transitions': dict(sorted(transition_frequency.items(), 
                                                key=lambda x: x[1], reverse=True)[:10]),
                'total_location_changes': len(transitions),
                'activity_consistency': self._calculate_consistency_score(locations)
            }
        
        return {}
    # location pattern
    def _analyze_location_patterns(self, entity_id):
        activities = self.entity_activities[entity_id]
        location_data = []
        
        for activity_type, activity_list in activities.items():
            for activity in activity_list:
                timestamp = activity.get('timestamp')
                location = self._extract_location(activity['record'], activity_type)
                
                if timestamp and location:
                    location_data.append({
                        'location': location,
                        'timestamp': timestamp,
                        'activity_type': activity_type,
                        'hour': timestamp.hour,
                        'day_part': self._get_day_part(timestamp.hour)
                    })
        
        if not location_data:
            return {}
        
        # Location preferences by time of day
        location_preferences = defaultdict(lambda: defaultdict(int))
        for data in location_data:
            location_preferences[data['day_part']][data['location']] += 1
        
        # Convert to percentages
        location_preferences_pct = {}
        for day_part, locations in location_preferences.items():
            total = sum(locations.values())
            location_preferences_pct[day_part] = {
                loc: count/total for loc, count in locations.items()
            }
        
        return {
            'location_preferences_by_time': location_preferences_pct,
            'most_visited_location': max(set([d['location'] for d in location_data]), 
                                       key=[d['location'] for d in location_data].count),
            'visit_frequency': len(location_data),
            'location_entropy': self._calculate_location_entropy([d['location'] for d in location_data])
        }
    # temporal pattern
    def _analyze_temporal_patterns(self, entity_id):
        activities = self.entity_activities[entity_id]
        temporal_data = []
        
        for activity_type, activity_list in activities.items():
            for activity in activity_list:
                timestamp = activity.get('timestamp')
                if timestamp:
                    temporal_data.append({
                        'hour': timestamp.hour,
                        'day_of_week': timestamp.weekday(),
                        'is_weekend': timestamp.weekday() >= 5,
                        'activity_type': activity_type
                    })
        
        if not temporal_data:
            return {}
        
        # Activity distribution by hour
        hourly_activity = defaultdict(int)
        for data in temporal_data:
            hourly_activity[data['hour']] += 1
        
        # Peak activity times
        peak_hours = sorted(hourly_activity.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'hourly_activity_distribution': dict(hourly_activity),
            'peak_activity_hours': [hour for hour, count in peak_hours],
            'weekday_vs_weekend_ratio': len([d for d in temporal_data if d['is_weekend']]) / len(temporal_data),
            'most_active_day': max(set([d['day_of_week'] for d in temporal_data]), 
                                 key=[d['day_of_week'] for d in temporal_data].count)
        }
    # generate evidence for inference
    def _generate_evidence_chains(self, entity_id):
        evidence_chains = []
        
        # Add cross-source links as evidence
        for cross_link in self.cross_source_links.get(entity_id, []):
            evidence_chains.append({
                'type': 'cross_source_correlation',
                'timestamp': cross_link['timestamp'].isoformat() if cross_link.get('timestamp') else None,
                'sources': cross_link.get('sources', []),
                'confidence': cross_link.get('confidence', 0),
                'description': cross_link.get('description', ''),
                'evidence_type': 'temporal_proximity'
            })
        
        # Add sequential evidence
        activities = self._get_all_timestamped_activities(entity_id)
        if len(activities) >= 2:
            activities.sort(key=lambda x: x['timestamp'])
            recent_sequence = [self._extract_location(act['record'], act['source']) 
                             for act in activities[-3:] if self._extract_location(act['record'], act['source'])]
            
            if len(recent_sequence) >= 2:
                evidence_chains.append({
                    'type': 'sequential_pattern',
                    'sequence': recent_sequence,
                    'confidence': 0.8,
                    'description': f"Recent movement pattern: {' → '.join(recent_sequence)}",
                    'evidence_type': 'behavioral_sequence'
                })
        
        return evidence_chains
    # featuring for ml 
    def _extract_ml_features(self, entity_id):
        activities = self.entity_activities[entity_id]
        profile = self.entity_registry[entity_id]
        
        # Basic features
        features = {
            'entity_id': entity_id,
            'department': profile.get('department', 'Unknown'),
            'role': profile.get('role', 'Unknown'),
            'total_activities': sum(len(act_list) for act_list in activities.values()),
            'activity_variety': len(activities),
            'data_sources_used': list(activities.keys())
        }
        
        # Location-based features
        locations = []
        for activity_type, activity_list in activities.items():
            for activity in activity_list:
                location = self._extract_location(activity['record'], activity_type)
                if location:
                    locations.append(location)
        
        if locations:
            features.update({
                'unique_locations_count': len(set(locations)),
                'most_frequent_location': max(set(locations), key=locations.count),
                'location_consistency': len(set(locations)) / len(locations) if locations else 0
            })
        
        return features
    # kepping only essential fields to reduce size
    def _clean_activity_details(self, record, activity_type):
        """Clean activity details for JSON output"""
        cleaned = {}
        
        keep_fields = {
            'wifi_logs': ['device_hash', 'ap_id', 'timestamp'],
            'campus_swipes': ['card_id', 'location_id', 'timestamp'],
            'library_checkouts': ['book_id', 'timestamp'],
            'lab_bookings': ['room_id', 'start_time', 'end_time'],
            'text_notes': ['category', 'text', 'timestamp'],
            'face_vectors': ['face_id', 'timestamp'],
            'cctv_frames': ['frame_id', 'location_id', 'timestamp']
        }
        
        fields_to_keep = keep_fields.get(activity_type, [])
        for field in fields_to_keep:
            if field in record:
                cleaned[field] = record[field]
        
        return cleaned
    # conv hour to day
    def _get_day_part(self, hour):
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    # cal how consistent location patterns are::
    def _calculate_consistency_score(self, locations):
        if len(locations) <= 1:
            return 1.0
        
        unique_locations = set(locations)
        if len(unique_locations) == 1:
            return 1.0
        
        return 1.0 - (len(unique_locations) / len(locations))
    # calc location distribution
    def _calculate_location_entropy(self, locations):
        from collections import Counter
        if not locations:
            return 0
        
        counter = Counter(locations)
        total = len(locations)
        entropy = 0
        
        for count in counter.values():
            p = count / total
            entropy -= p * np.log2(p)
        
        return entropy
    # check our linking success or not
    def _calculate_linking_success_rate(self):
        total_records = 0
        linked_records = 0
        
        for dataset_name, df in self.datasets.items():
            if dataset_name != 'profile':  # Skip profiles
                total_records += len(df)
        
        linked_records = self._count_total_activities()
        
        return linked_records / total_records if total_records > 0 else 0
    # assess how good the patterns ar for ml
    def _assess_pattern_richness(self):
        total_activities = self._count_total_activities()
        total_entities = len(self.entity_registry)
        
        if total_activities == 0:
            return "low"
        
        activities_per_entity = total_activities / total_entities
        
        if activities_per_entity > 20:
            return "high"
        elif activities_per_entity > 10:
            return "medium"
        else:
            return "low"
    
    def _assess_temporal_coverage(self):
        """Assess temporal coverage of the data"""
        return "requires_timestamp_analysis"
    
    def _assess_data_completeness(self):
        """Assess data completeness for prediction"""
        entities_with_activities = sum(1 for entity_acts in self.entity_activities.values() 
                                     if any(entity_acts.values()))
        completeness = entities_with_activities / len(self.entity_registry)
        
        if completeness > 0.8:
            return "high"
        elif completeness > 0.5:
            return "medium"
        else:
            return "low"

#generate and save final entity resolver json file
def generate_enhanced_json_output():    
    # Load datasets
    datasets = load_all_datasets()
    if not datasets:
        return None
    
    # calling base class
    resolver = ImprovedEntityResolver(datasets)
    resolver.resolve_all_entities_full_pipeline()
    
    # Generate JSON
    enhanced_output = resolver.generate_enhanced_json_output()
    
    # Save to file
    output_filename = f"Entity_resolution_map1.json"
    with open(output_filename, 'w') as f:
        json.dump(enhanced_output, f, indent=2, default=str)
    
    
    return enhanced_output

if __name__ == "__main__":
    enhanced_json = generate_enhanced_json_output()
    

        

        
