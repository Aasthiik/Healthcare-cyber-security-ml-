"""
Advanced Analytics and Real-time Monitoring Dashboard
Provides comprehensive threat analytics and visualization data

Usage: Integrated into Flask app.py
"""

import logging
import json
from datetime import datetime, timedelta
import sqlite3
import numpy as np
from collections import defaultdict, Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedAnalytics:
    """Comprehensive threat analytics and reporting"""
    
    def __init__(self, db_path='analytics.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize analytics database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('PRAGMA journal_mode=WAL')
            
            # Threat events table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS threat_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    threat_type TEXT NOT NULL,
                    severity INTEGER NOT NULL,
                    source_ip TEXT,
                    destination_ip TEXT,
                    confidence_score REAL NOT NULL,
                    anomaly_score REAL,
                    model_name TEXT,
                    user_id INTEGER,
                    session_id TEXT
                )
            ''')
            
            # Prediction metrics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS prediction_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model_name TEXT NOT NULL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    inference_time_ms REAL,
                    sample_count INTEGER
                )
            ''')
            
            # System health table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    cpu_usage REAL,
                    memory_usage REAL,
                    active_sessions INTEGER,
                    total_predictions_today INTEGER,
                    model_status TEXT
                )
            ''')
            
            # Create indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_threat_timestamp ON threat_events(timestamp DESC)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_threat_type ON threat_events(threat_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_threat_severity ON threat_events(severity DESC)')
            
            conn.commit()
            conn.close()
            logger.info("✅ Analytics database initialized")
            
        except Exception as e:
            logger.error(f"❌ Database initialization error: {e}")
    
    def log_threat_event(self, threat_type, severity, source_ip, dest_ip, 
                        confidence, anomaly_score, model_name, user_id=None):
        """Log a threat detection event"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT INTO threat_events 
                (threat_type, severity, source_ip, destination_ip, confidence_score, 
                 anomaly_score, model_name, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (threat_type, severity, source_ip, dest_ip, confidence, anomaly_score, model_name, user_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging threat event: {e}")
    
    def log_prediction_metrics(self, model_name, accuracy, precision, recall, f1, inference_time_ms):
        """Log model performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT INTO prediction_metrics 
                (model_name, accuracy, precision, recall, f1_score, inference_time_ms, sample_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (model_name, accuracy, precision, recall, f1, inference_time_ms, 1))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
    
    def get_threat_statistics(self, hours=24):
        """Get threat statistics for specified time period"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            # Threats by type
            cursor = conn.execute('''
                SELECT threat_type, COUNT(*) as count, AVG(severity) as avg_severity
                FROM threat_events
                WHERE timestamp > ?
                GROUP BY threat_type
                ORDER BY count DESC
            ''', (cutoff_time,))
            
            threats_by_type = [dict(row) for row in cursor.fetchall()]
            
            # Threats by severity
            cursor = conn.execute('''
                SELECT severity, COUNT(*) as count
                FROM threat_events
                WHERE timestamp > ?
                GROUP BY severity
                ORDER BY severity DESC
            ''', (cutoff_time,))
            
            threats_by_severity = [dict(row) for row in cursor.fetchall()]
            
            # Top source IPs
            cursor = conn.execute('''
                SELECT source_ip, COUNT(*) as count, AVG(severity) as avg_severity
                FROM threat_events
                WHERE timestamp > ? AND source_ip IS NOT NULL
                GROUP BY source_ip
                ORDER BY count DESC
                LIMIT 10
            ''', (cutoff_time,))
            
            top_source_ips = [dict(row) for row in cursor.fetchall()]
            
            # Total stats
            cursor = conn.execute('''
                SELECT 
                    COUNT(*) as total_events,
                    AVG(confidence_score) as avg_confidence,
                    MAX(severity) as max_severity,
                    COUNT(DISTINCT source_ip) as unique_sources
                FROM threat_events
                WHERE timestamp > ?
            ''', (cutoff_time,))
            
            total_stats = dict(cursor.fetchone())
            
            conn.close()
            
            return {
                'threats_by_type': threats_by_type,
                'threats_by_severity': threats_by_severity,
                'top_source_ips': top_source_ips,
                'total_stats': total_stats,
                'time_period_hours': hours
            }
        
        except Exception as e:
            logger.error(f"Error getting threat statistics: {e}")
            return None
    
    def get_model_performance(self, hours=24):
        """Get model performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            cursor = conn.execute('''
                SELECT 
                    model_name,
                    AVG(accuracy) as avg_accuracy,
                    AVG(precision) as avg_precision,
                    AVG(recall) as avg_recall,
                    AVG(f1_score) as avg_f1,
                    AVG(inference_time_ms) as avg_inference_time,
                    COUNT(*) as sample_count
                FROM prediction_metrics
                WHERE timestamp > ?
                GROUP BY model_name
                ORDER BY avg_f1 DESC
            ''', (cutoff_time,))
            
            models = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return models
        
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return []
    
    def get_threat_timeline(self, hours=24, limit=100):
        """Get timeline of recent threats"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            cursor = conn.execute('''
                SELECT 
                    timestamp, threat_type, severity, source_ip, 
                    confidence_score, model_name
                FROM threat_events
                WHERE timestamp > ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (cutoff_time, limit))
            
            events = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return events
        
        except Exception as e:
            logger.error(f"Error getting threat timeline: {e}")
            return []
    
    def get_threat_trends(self, days=7):
        """Get threat trends over time"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            
            cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor = conn.execute('''
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as daily_count,
                    AVG(severity) as avg_severity
                FROM threat_events
                WHERE timestamp > ?
                GROUP BY DATE(timestamp)
                ORDER BY date ASC
            ''', (cutoff_time,))
            
            trends = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return trends
        
        except Exception as e:
            logger.error(f"Error getting threat trends: {e}")
            return []
    
    def get_anomaly_analysis(self, hours=24):
        """Analyze anomalies detected in time period"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            cursor = conn.execute('''
                SELECT 
                    threat_type,
                    AVG(anomaly_score) as avg_anomaly_score,
                    MAX(anomaly_score) as max_anomaly_score,
                    COUNT(*) as count
                FROM threat_events
                WHERE timestamp > ? AND anomaly_score IS NOT NULL
                GROUP BY threat_type
                ORDER BY avg_anomaly_score DESC
            ''', (cutoff_time,))
            
            anomalies = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return anomalies
        
        except Exception as e:
            logger.error(f"Error analyzing anomalies: {e}")
            return []
    
    def generate_security_report(self, hours=24):
        """Generate comprehensive security report"""
        try:
            report = {
                'generated_at': datetime.now().isoformat(),
                'period_hours': hours,
                'threat_statistics': self.get_threat_statistics(hours),
                'model_performance': self.get_model_performance(hours),
                'threat_trends': self.get_threat_trends(7),
                'anomaly_analysis': self.get_anomaly_analysis(hours),
                'recent_threats': self.get_threat_timeline(hours, limit=20)
            }
            
            return report
        
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return None


# Global instance
analytics = None


def init_analytics(db_path='analytics.db'):
    """Initialize analytics system"""
    global analytics
    analytics = AdvancedAnalytics(db_path)
    return analytics


def get_analytics():
    """Get analytics instance"""
    global analytics
    if analytics is None:
        analytics = init_analytics()
    return analytics
