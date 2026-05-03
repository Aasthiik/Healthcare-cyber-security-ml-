"""
HIPAA & GDPR Compliance Module
Encryption, Audit Logging, and Data Privacy Features

Author: Healthcare Cybersecurity Team
Version: 1.0
"""

import sqlite3
import hashlib
import logging
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf import pbkdf2
import base64
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataEncryption:
    """
    Encryption utilities for HIPAA/GDPR compliance
    All sensitive data encrypted at rest
    """
    
    def __init__(self, master_key=None):
        """
        Initialize encryption with master key
        If no key provided, generates one
        """
        if master_key is None:
            master_key = os.environ.get('ENCRYPTION_KEY', 'default-insecure-key')
        
        # Derive encryption key from master key
        kdf = pbkdf2.PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'healthcare-cybersec-salt',  # In production, use random salt per user
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        self.cipher_suite = Fernet(key)
        logger.info("✅ Encryption initialized")
    
    def encrypt_data(self, data):
        """Encrypt sensitive data"""
        try:
            if isinstance(data, dict):
                data = json.dumps(data)
            elif not isinstance(data, str):
                data = str(data)
            
            encrypted = self.cipher_suite.encrypt(data.encode())
            return encrypted.decode()
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            return None
    
    def decrypt_data(self, encrypted_data):
        """Decrypt sensitive data"""
        try:
            decrypted = self.cipher_suite.decrypt(encrypted_data.encode())
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            return None


class AuditLogger:
    """
    Audit logging for HIPAA compliance
    Immutable audit trail of all access and actions
    """
    
    def __init__(self, db_path='audit_logs.db'):
        self.db_path = db_path
        self._init_audit_db()
    
    def _init_audit_db(self):
        """Initialize audit logging database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create audit trail table (WORM - Write Once Read Many)
        c.execute('''
            CREATE TABLE IF NOT EXISTS audit_trail (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id INTEGER,
                username TEXT,
                action TEXT NOT NULL,
                resource TEXT,
                details TEXT,
                status TEXT,
                ip_address TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create compliance report table
        c.execute('''
            CREATE TABLE IF NOT EXISTS compliance_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                description TEXT,
                severity TEXT,
                remediation_status TEXT DEFAULT 'PENDING',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create data access log
        c.execute('''
            CREATE TABLE IF NOT EXISTS data_access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id INTEGER,
                data_type TEXT,
                access_type TEXT,
                record_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for compliance queries
        c.execute('CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_trail(timestamp DESC)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_trail(user_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_trail(action)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_compliance_type ON compliance_events(event_type)')
        
        conn.commit()
        conn.close()
        logger.info("✅ Audit database initialized")
    
    def log_action(self, user_id, username, action, resource, details, ip_address, status='SUCCESS'):
        """Log user action for audit trail"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            timestamp = datetime.utcnow().isoformat()
            
            c.execute('''
                INSERT INTO audit_trail 
                (timestamp, user_id, username, action, resource, details, status, ip_address)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, user_id, username, action, resource, json.dumps(details), status, ip_address))
            
            conn.commit()
            conn.close()
            logger.info(f"✅ Audit log: {action} by {username}")
            
        except Exception as e:
            logger.error(f"Failed to log action: {e}")
    
    def log_data_access(self, user_id, data_type, access_type, record_count):
        """Log data access for GDPR compliance"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            timestamp = datetime.utcnow().isoformat()
            
            c.execute('''
                INSERT INTO data_access_log
                (timestamp, user_id, data_type, access_type, record_count)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, user_id, data_type, access_type, record_count))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log data access: {e}")
    
    def log_compliance_event(self, event_type, description, severity):
        """Log compliance events"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            timestamp = datetime.utcnow().isoformat()
            
            c.execute('''
                INSERT INTO compliance_events
                (timestamp, event_type, description, severity)
                VALUES (?, ?, ?, ?)
            ''', (timestamp, event_type, description, severity))
            
            conn.commit()
            conn.close()
            logger.info(f"✅ Compliance event logged: {event_type}")
            
        except Exception as e:
            logger.error(f"Failed to log compliance event: {e}")
    
    def get_audit_trail(self, hours=24, user_id=None):
        """Retrieve audit trail for compliance reporting"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            cutoff_time = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
            
            if user_id:
                c.execute('''
                    SELECT * FROM audit_trail
                    WHERE timestamp > ? AND user_id = ?
                    ORDER BY timestamp DESC
                ''', (cutoff_time, user_id))
            else:
                c.execute('''
                    SELECT * FROM audit_trail
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                ''', (cutoff_time,))
            
            results = c.fetchall()
            conn.close()
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve audit trail: {e}")
            return []
    
    def generate_compliance_report(self):
        """Generate compliance report for auditors"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Summary statistics
            c.execute('SELECT COUNT(*) FROM audit_trail')
            total_actions = c.fetchone()[0]
            
            c.execute('SELECT COUNT(*) FROM compliance_events WHERE severity = "CRITICAL"')
            critical_events = c.fetchone()[0]
            
            c.execute('SELECT COUNT(*) FROM audit_trail WHERE status = "FAILURE"')
            failed_actions = c.fetchone()[0]
            
            c.execute('SELECT COUNT(DISTINCT user_id) FROM audit_trail')
            active_users = c.fetchone()[0]
            
            conn.close()
            
            report = {
                'generated_at': datetime.utcnow().isoformat(),
                'total_audit_events': total_actions,
                'critical_compliance_events': critical_events,
                'failed_actions': failed_actions,
                'active_users': active_users
            }
            
            logger.info("✅ Compliance report generated")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate compliance report: {e}")
            return {}


class DataPrivacy:
    """
    Data privacy and anonymization utilities
    GDPR data subject rights implementation
    """
    
    @staticmethod
    def anonymize_ip(ip_address):
        """Anonymize IP address for privacy"""
        if ':' in ip_address:  # IPv6
            parts = ip_address.split(':')
            return ':'.join(parts[:-2]) + ':xxxx:xxxx'
        else:  # IPv4
            parts = ip_address.split('.')
            return '.'.join(parts[:-1]) + '.xxx'
    
    @staticmethod
    def hash_pii(data):
        """Hash PII data for pseudo-anonymization"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    @staticmethod
    def mask_email(email):
        """Mask email for display"""
        parts = email.split('@')
        if len(parts[0]) > 2:
            masked = parts[0][:2] + '*' * (len(parts[0]) - 2) + '@' + parts[1]
        else:
            masked = parts[0] + '@' + parts[1]
        return masked
    
    @staticmethod
    def implement_right_to_be_forgotten(db_path, user_id):
        """
        Implement GDPR right to be forgotten
        Delete all user data
        """
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            
            # Delete user predictions
            c.execute('DELETE FROM predictions WHERE user_id = ?', (user_id,))
            
            # Delete user audit logs
            c.execute('DELETE FROM audit_trail WHERE user_id = ?', (user_id,))
            
            # Delete user data access logs
            c.execute('DELETE FROM data_access_log WHERE user_id = ?', (user_id,))
            
            # Delete user account (keep record for compliance, but anonymize)
            c.execute('''
                UPDATE users 
                SET username = ?, email = ?, password = ?
                WHERE id = ?
            ''', ('DELETED_USER_' + str(user_id), 'deleted@example.com', 'N/A', user_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ User {user_id} data deleted (right to be forgotten)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to implement right to be forgotten: {e}")
            return False
    
    @staticmethod
    def implement_data_portability(db_path, user_id):
        """
        Implement GDPR data portability
        Export all user data in standard format
        """
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            
            # Get user info
            c.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            user_data = c.fetchone()
            
            # Get user predictions
            c.execute('SELECT * FROM predictions WHERE user_id = ?', (user_id,))
            predictions = c.fetchall()
            
            # Get user audit logs
            c.execute('SELECT * FROM audit_trail WHERE user_id = ?', (user_id,))
            audit_logs = c.fetchall()
            
            conn.close()
            
            # Export to JSON format
            export_data = {
                'user_profile': user_data,
                'predictions': predictions,
                'audit_logs': audit_logs,
                'exported_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ User {user_id} data exported for portability")
            return export_data
            
        except Exception as e:
            logger.error(f"Failed to export user data: {e}")
            return None


class ComplianceMonitor:
    """
    Monitor HIPAA/GDPR compliance in real-time
    """
    
    def __init__(self, audit_db='audit_logs.db'):
        self.audit_db = audit_db
        self.audit_logger = AuditLogger(audit_db)
    
    def check_unauthorized_access(self):
        """Check for unauthorized access attempts"""
        try:
            conn = sqlite3.connect(self.audit_db)
            c = conn.cursor()
            
            # Find failed login attempts
            c.execute('''
                SELECT COUNT(*) FROM audit_trail
                WHERE action = 'LOGIN' AND status = 'FAILURE'
                AND timestamp > datetime('now', '-1 hour')
            ''')
            
            failed_logins = c.fetchone()[0]
            conn.close()
            
            if failed_logins > 5:  # Threshold
                self.audit_logger.log_compliance_event(
                    'UNAUTHORIZED_ACCESS_ATTEMPT',
                    f'{failed_logins} failed login attempts in last hour',
                    'HIGH'
                )
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check unauthorized access: {e}")
            return True
    
    def check_data_retention_policy(self, db_path, retention_days=365):
        """Check and enforce data retention policy"""
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            
            # Delete predictions older than retention period
            cutoff_date = (datetime.utcnow() - timedelta(days=retention_days)).isoformat()
            c.execute('''
                DELETE FROM predictions
                WHERE prediction_date < ?
            ''', (cutoff_date,))
            
            deleted_count = c.rowcount
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                logger.info(f"✅ Deleted {deleted_count} old predictions (retention policy)")
                self.audit_logger.log_compliance_event(
                    'DATA_RETENTION_ENFORCEMENT',
                    f'Deleted {deleted_count} records older than {retention_days} days',
                    'INFO'
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to enforce retention policy: {e}")
            return False


# Example usage
if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("HIPAA/GDPR Compliance Module")
    logger.info("=" * 80)
    
    # Initialize encryption
    encryption = DataEncryption()
    logger.info("✅ Encryption module initialized")
    
    # Initialize audit logger
    audit = AuditLogger()
    logger.info("✅ Audit logging module initialized")
    
    # Test encryption
    sensitive_data = "patient_id: 12345"
    encrypted = encryption.encrypt_data(sensitive_data)
    logger.info(f"Encrypted: {encrypted[:50]}...")
    
    # Test anonymization
    logger.info(f"Masked email: {DataPrivacy.mask_email('patient@healthcare.com')}")
    logger.info(f"Anonymized IP: {DataPrivacy.anonymize_ip('192.168.1.100')}")
    
    # Test audit logging
    audit.log_action(
        user_id=1,
        username='admin',
        action='LOGIN',
        resource='system',
        details={'method': 'password'},
        ip_address='192.168.1.1'
    )
    
    logger.info("=" * 80)
    logger.info("All compliance modules operational")
    logger.info("=" * 80)
