"""
Alarm System for SentiGuard
============================
Manages alarm triggers, SMS notifications, and alarm state
"""
import time
import os
from datetime import datetime
from typing import Optional

# SMS Integration (Twilio)
try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    print("⚠️ Twilio not installed. SMS notifications disabled.")
    print("   Install with: pip install twilio")

# Database integration (optional)
try:
    from database import async_session_maker
    from models import AlarmEvent
    DATABASE_AVAILABLE = True
except:
    DATABASE_AVAILABLE = False

class AlarmSystem:
    def __init__(self):
        self.alarm_active = False
        self.alarm_start_time = None
        self.alarm_duration = 5.0  # seconds
        self.cooldown_period = 30.0  # seconds between alarms
        self.last_alarm_time = 0
        self.alarm_history = []
        
        # SMS configuration
        self.sms_enabled = False
        self.twilio_client = None
        self.load_sms_config()
        
    def load_sms_config(self):
        """Load SMS configuration from environment or config file"""
        account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.from_number = os.getenv('TWILIO_FROM_NUMBER')
        self.to_numbers = os.getenv('TWILIO_TO_NUMBERS', '').split(',')
        
        if TWILIO_AVAILABLE and account_sid and auth_token and self.from_number:
            try:
                self.twilio_client = Client(account_sid, auth_token)
                self.sms_enabled = True
                print(f"✓ SMS notifications enabled: {self.from_number}")
            except Exception as e:
                print(f"❌ Twilio initialization failed: {e}")
        else:
            print("ℹ️ SMS not configured (set TWILIO_* environment variables)")
    
    def check_and_trigger(self, threat_score: float, status: str) -> dict:
        """
        Check if alarm should trigger and manage state
        
        Returns:
            dict with alarm_active, alarm_sound_play, sms_sent
        """
        current_time = time.time()
        result = {
            "alarm_active": False,
            "alarm_sound_play": False,
            "sms_sent": False,
            "message": None
        }
        
        # Check if in cooldown
        time_since_last = current_time - self.last_alarm_time
        if time_since_last < self.cooldown_period:
            if self.alarm_active:
                result["alarm_active"] = True
            return result
        
        # Trigger condition: threat >= 60%
        if threat_score >= 0.60:
            if not self.alarm_active:
                # NEW ALARM!
                self.alarm_active = True
                self.alarm_start_time = current_time
                self.last_alarm_time = current_time
                
                result["alarm_active"] = True
                result["alarm_sound_play"] = True
                
                # Send SMS
                if self.sms_enabled:
                    sms_result = self.send_sms_alert(threat_score, status)
                    result["sms_sent"] = sms_result
                
                # Log alarm
                self.alarm_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "threat_score": threat_score,
                    "status": status
                })
                
                # TODO: Database save requires async - will implement later
                # Save to database if available
                # if DATABASE_AVAILABLE:
                #     await save_alarm_to_db(alarm_event)
                
                print(f"🚨 ALARM TRIGGERED! Threat: {threat_score*100:.1f}%")
                result["message"] = "ALARM ACTIVATED"
            else:
                # Alarm still active
                result["alarm_active"] = True
        else:
            # Check if alarm should deactivate
            if self.alarm_active:
                elapsed = current_time - self.alarm_start_time
                if elapsed >= self.alarm_duration:
                    self.alarm_active = False
                    print("✓ Alarm deactivated (5s elapsed)")
                else:
                    result["alarm_active"] = True
        
        return result
    
    def send_sms_alert(self, threat_score: float, status: str) -> bool:
        """Send SMS alert to configured numbers"""
        if not self.sms_enabled or not self.twilio_client:
            return False
        
        message_body = f"🚨 SENTIGUARD ALERT!\n\nThreat Level: {threat_score*100:.0f}%\nStatus: {status}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nImmediate attention required!"
        
        sent_count = 0
        for to_number in self.to_numbers:
            to_number = to_number.strip()
            if not to_number:
                continue
            try:
                message = self.twilio_client.messages.create(
                    body=message_body,
                    from_=self.from_number,
                    to=to_number
                )
                print(f"📱 SMS sent to {to_number}: {message.sid}")
                sent_count += 1
            except Exception as e:
                print(f"❌ SMS failed to {to_number}: {e}")
        
        return sent_count > 0
    
    def get_history(self, limit: int = 10):
        """Get recent alarm history"""
        return self.alarm_history[-limit:]
    
    def test_alarm(self) -> dict:
        """Test alarm system"""
        return self.check_and_trigger(0.65, "TEST_ALARM")

# Global instance
alarm_system = AlarmSystem()
