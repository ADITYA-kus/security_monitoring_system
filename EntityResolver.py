import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
import json

class CompleteEntityResolver:
    def __init__(self, datasets):
        self.datasets = datasets
        self.entity_registry = {}
        self.id_to_entity = {}
        self.entity_activities = defaultdict(lambda: defaultdict(list))
        self.cross_source_links = defaultdict(list)
        self.confidence_scores = {}
    # pipeline for entity resolution    
    def resolve_all_entities_full_pipeline(self):
        
        # Build entity maps from ALL profiles
        self._build_complete_entity_maps()
        
        # Link ALL activities across ALL datasets
        self._link_all_data_sources()
        
        # Create cross-source relationships
        self._create_inferred_relationships()
        
        # Multi-modal fusion
        self._perform_multi_modal_fusion()
        
        # make comprehensive output and dump on json
        clean_data = self._generate_clean_output()        
        return clean_data
    
    # start with entity maping
    def _build_complete_entity_maps(self):        
        if 'profile' not in self.datasets:
            raise ValueError("Profile dataset not found")
        
        total_profiles = len(self.datasets['profile'])
        print(f"Processing {total_profiles} profiles")
        
        for idx, person in self.datasets['profile'].iterrows():
            entity_id = person['entity_id']
            
            # identifiers
            identifiers = []
            id_fields = ['student_id', 'staff_id', 'card_id', 'device_hash', 'face_id', 'email']
            
            for field in id_fields:
                if field in person and pd.notna(person[field]):
                    identifiers.append(str(person[field]))
            
            # Map ALL identifiers to this entity
            for identifier in identifiers:
                self.id_to_entity[identifier] = entity_id
            
            # Store complete entity info
            self.entity_registry[entity_id] = {
                'name': person.get('name', 'Unknown'),
                'role': person.get('role', 'Unknown'),
                'email': person.get('email', ''),
                'department': person.get('department', ''),
                'all_identifiers': identifiers,
                'source': 'profile',
                'resolution_method': 'direct_mapping'
            }
            
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
                print(f"Processing {len(df)} records from {dataset_name}")
                
                linked_count = 0
                for _, record in df.iterrows():
                    identifier = record.get(id_field)
                    entity_id = self._find_entity(identifier, id_field)
                    
                    if entity_id:
                        self.entity_activities[entity_id][activity_type].append({
                            'record': record.to_dict(),
                            'source': dataset_name,
                            'confidence': 1.0,
                            'provenance': f"direct_{id_field}_match",
                            'timestamp': self._extract_timestamp(record, activity_type)
                        })
                        linked_count += 1
                
                total_linked += linked_count
                print(f"Linked {linked_count}/{len(df)} records from {dataset_name}")
        
        print(f"Total linked activities: {total_linked}")
    
    def _create_inferred_relationships(self):
        cross_link_count = 0
        for entity_id in self.entity_registry.keys():
            # Get ALL activities for this entity
            all_activities = self._get_all_timestamped_activities(entity_id)
            
            # Group by time windows (activities within 30 minutes)
            time_groups = self._group_by_time_windows(all_activities)
            
            # Create cross-source links for each time group
            for time_group in time_groups:
                if len(time_group) >= 2:  
                    self._create_cross_source_evidence(entity_id, time_group)
                    cross_link_count += 1
        
        print(f" Created {cross_link_count} cross-source relationships")
    
    def _perform_multi_modal_fusion(self):        
        for entity_id in self.entity_registry.keys():
            # Collect evidence from ALL sources
            evidence = self._collect_all_evidence_types(entity_id)
            
            # Calculate overall confidence score
            final_confidence = self._calculate_fused_confidence(evidence)
            
            self.confidence_scores[entity_id] = {
                'final_confidence': final_confidence,
                'evidence_breakdown': {
                    'source_count': len(evidence),
                    'activity_count': sum(len(activities) for activities in self.entity_activities[entity_id].values()),
                    'cross_links_count': len(self.cross_source_links.get(entity_id, []))
                },
                'provenance': 'multi_modal_fusion'
            }
            
    def _generate_clean_output(self):
        
        clean_output = {
            'entities': {},
            'statistics': {
                'total_entities': len(self.entity_registry),
                'total_activities': self._count_total_activities(),
                'total_cross_links': self._count_cross_links(),
                'data_sources_used': list(self.datasets.keys()),
                'resolution_timestamp': datetime.now().isoformat()
            },
            'patterns_ready': True
        }
        
        # Structure data for easy pattern analysis
        for entity_id in self.entity_registry.keys():
            entity_data = {
                'profile': self.entity_registry[entity_id],
                'activities': self._get_structured_activities(entity_id),
                'cross_source_evidence': self.cross_source_links.get(entity_id, []),
                'confidence': self.confidence_scores.get(entity_id, {}),
                'behavioral_summary': self._generate_behavioral_summary(entity_id)
            }
            clean_output['entities'][entity_id] = entity_data
        
        return clean_output
    
    def _get_structured_activities(self, entity_id):
        """Get all activities in structured format for pattern analysis"""
        structured = {}
        
        for activity_type, activities in self.entity_activities[entity_id].items():
            structured[activity_type] = []
            for activity in activities:
                structured[activity_type].append({
                    'timestamp': activity['timestamp'].isoformat() if activity['timestamp'] else None,
                    'location': self._extract_location(activity['record'], activity_type),
                    'details': activity['record'],
                    'confidence': activity['confidence'],
                    'source': activity['source']
                })
        
        return structured
    
    def _generate_behavioral_summary(self, entity_id):
        """Generate behavioral summary for pattern recognition"""
        activities = self.entity_activities[entity_id]
        
        # Extract location sequences
        locations = []
        timestamps = []
        
        for activity_type, activity_list in activities.items():
            for activity in activity_list:
                if activity['timestamp']:
                    location = self._extract_location(activity['record'], activity_type)
                    if location:
                        locations.append(location)
                        timestamps.append(activity['timestamp'])
        
        # Sort by timestamp
        if timestamps:
            sorted_data = sorted(zip(timestamps, locations))
            locations_sequence = [loc for _, loc in sorted_data]
        else:
            locations_sequence = []
        
        return {
            'total_activities': sum(len(act_list) for act_list in activities.values()),
            'unique_locations': len(set(locations)),
            'location_sequence': locations_sequence,
            'activity_types': list(activities.keys()),
            'time_range': {
                'first_activity': min(timestamps).isoformat() if timestamps else None,
                'last_activity': max(timestamps).isoformat() if timestamps else None
            }
        }
    
    #  UTILITY METHOD
    def _find_entity(self, identifier, id_field):
        """Find entity by identifier with validation"""
        if pd.isna(identifier):
            return None
        
        entity_id = self.id_to_entity.get(str(identifier))
        return entity_id
    
    def _get_all_timestamped_activities(self, entity_id):
        """Get all activities with timestamps for an entity"""
        all_activities = []
        
        for activity_type, activities in self.entity_activities[entity_id].items():
            for activity in activities:
                if activity['timestamp']:
                    all_activities.append(activity)
        
        return all_activities
    
    def _group_by_time_windows(self, activities, window_minutes=30):
        """Group activities by time windows"""
        if not activities:
            return []
        
        time_groups = []
        sorted_activities = sorted(activities, key=lambda x: x['timestamp'])
        
        for activity in sorted_activities:
            placed = False
            for group in time_groups:
                time_diff = abs((activity['timestamp'] - group[0]['timestamp']).total_seconds() / 60)
                if time_diff <= window_minutes:
                    group.append(activity)
                    placed = True
                    break
            
            if not placed:
                time_groups.append([activity])
        
        return time_groups
    
    def _create_cross_source_evidence(self, entity_id, related_activities):
        """Create cross-source evidence chain"""
        sources = list(set(act['source'] for act in related_activities))
        
        cross_link = {
            'type': 'temporal_correlation',
            'sources': sources,
            'activities': related_activities,
            'timestamp': related_activities[0]['timestamp'],
            'confidence': min(0.9, 0.7 + (len(sources) * 0.05)),  # More sources = higher confidence
            'provenance': 'inferred_temporal_pattern',
            'description': f"Activities from {len(sources)} sources within 30 minutes"
        }
        
        self.cross_source_links[entity_id].append(cross_link)
    
    def _extract_timestamp(self, record, activity_type):
        """Extract timestamp from record"""
        timestamp_fields = {
            'wifi_logs': 'timestamp',
            'campus_swipes': 'timestamp',
            'library_checkouts': 'timestamp',
            'lab_bookings': 'start_time',
            'text_notes': 'timestamp',
            'cctv_frames': 'timestamp',
            'face_vectors': 'timestamp'
        }
        
        field = timestamp_fields.get(activity_type)
        if field and field in record and pd.notna(record[field]):
            try:
                return pd.to_datetime(record[field])
            except:
                return None
        return None
    
    def _extract_location(self, record, activity_type):
        """Extract location from record"""
        if 'location_id' in record:
            return record['location_id']
        elif 'ap_id' in record:
            return record['ap_id']
        elif 'room_id' in record:
            return record['room_id']
        return None
    
    def _collect_all_evidence_types(self, entity_id):
        """Collect all evidence types for an entity"""
        evidence_types = set()
        
        for activity_type in self.entity_activities[entity_id].keys():
            evidence_types.add(activity_type)
        
        if self.cross_source_links.get(entity_id):
            evidence_types.add('cross_source')
        
        return list(evidence_types)
    
    def _calculate_fused_confidence(self, evidence_types):
        """Calculate fused confidence score"""
        if not evidence_types:
            return 0.0
        
        base_score = 0.7
        bonus_per_source = 0.05
        return min(0.95, base_score + (len(evidence_types) * bonus_per_source))
    
    def _count_total_activities(self):
        """Count total activities across all entities"""
        total = 0
        for entity_acts in self.entity_activities.values():
            for activity_list in entity_acts.values():
                total += len(activity_list)
        return total
    
    def _count_cross_links(self):
        """Count total cross-source links"""
        return sum(len(links) for links in self.cross_source_links.values())


def load_all_datasets():    
    datasets = {}
    try:
        datasets['profile'] = pd.read_csv('/content/student or staff profiles.csv')
        datasets['wifi_logs'] = pd.read_csv('/content/wifi_associations_logs.csv')
        datasets['campus_swipes'] = pd.read_csv('/content/campus card_swipes.csv')
        datasets['library_check'] = pd.read_csv('/content/library_checkouts.csv')
        datasets['lab_bookings'] = pd.read_csv('/content/lab_bookings.csv')
        datasets['text_notes'] = pd.read_csv('/content/free_text_notes (helpdesk or RSVPs).csv')
        datasets['face_vector'] = pd.read_csv('/content/face_embeddings.csv')
        datasets['cctv_frame'] = pd.read_csv('/content/cctv_frames.csv')
        
        print("All datasets loaded successfully")
        return datasets
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return {}

def run_complete_entity_resolution():
    # Load all data
    datasets = load_all_datasets()
    if not datasets:
        print("Failed to load datasets")
        return None
    
    # Initialize and run resolver
    resolver = CompleteEntityResolver(datasets)
    clean_data = resolver.resolve_all_entities_full_pipeline()
    return clean_data

# Run the complete pipeline
if __name__ == "__main__":
    clean_data = run_complete_entity_resolution()
    