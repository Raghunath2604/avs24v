import sounddevice as sd
import numpy as np
import time

class AudioDetector:
    def __init__(self):
        self.simulated_mode = False
        self.sample_rate = 44100
        self.duration = 1.0 # seconds per chunk
        self.stream = None
        self.last_score = 0.0
        self.last_trigger_time = 0
        self.last_rms = 0.0  # Track RMS for ambient noise
        
        # Start audio stream
        try:
            self.stream = sd.InputStream(callback=self.audio_callback, channels=1, samplerate=self.sample_rate)
            self.stream.start()
            print("Microphone started successfully.")
        except Exception as e:
            print(f"Error starting microphone: {e}. Falling back to simulation.")
            self.simulated_mode = True
            

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        # Compute RMS amplitude (Volume)
        volume_norm = np.linalg.norm(indata) * 10
        self.last_rms = volume_norm  # Store for ambient noise computation
        
        # Compute FFT for Frequency Analysis (Pitch)
        # We want to detect high-pitched screams (typically 500Hz - 3000Hz)
        try:
            # Apply Hanning window to reduce spectral leakage
            windowed_data = indata[:, 0] * np.hanning(len(indata))
            fft_data = np.fft.rfft(windowed_data)
            fft_freq = np.fft.rfftfreq(len(windowed_data), 1 / self.sample_rate)
            
            # Find dominant frequency
            magnitude = np.abs(fft_data)
            peak_freq = fft_freq[np.argmax(magnitude)]
            
            # Scream Logic:
            # 1. Must be loud (volume > threshold)
            # 2. Must be high pitched (freq > 400Hz) - Lowered from 800Hz for better testing
            
            is_loud = volume_norm > 2.0 # Lowered threshold significantly
            is_high_pitch = peak_freq > 400 and peak_freq < 4000
            
            # Debug logging (print every ~10th frame to avoid spam, or just print if loud)
            if is_loud:
                 print(f"🎤 Audio: Vol={volume_norm:.1f}, Freq={peak_freq:.0f}Hz")

            if is_loud and is_high_pitch:
                # High probability of scream
                self.last_score = min(1.0, (volume_norm / 20.0) + 0.5) # Boost score
            else:
                # Just loud noise or background - still count it!
                self.last_score = min(0.8, volume_norm / 40.0) 
                
        except Exception as e:
            print(f"Audio Error: {e}")
            self.last_score = min(1.0, volume_norm / 50.0) 

    def trigger_distress(self):
        """Manually trigger a distress signal for testing (fallback)."""
        if self.simulated_mode:
            self.last_trigger_time = time.time()

    def get_score(self):
        """Get vocal scream score [0..1]."""
        if self.simulated_mode:
            now = time.time()
            if now - self.last_trigger_time < 2.0:
                return 0.9
            return 0.0
        
        return float(self.last_score)
    
    def get_rms(self):
        """Get current RMS energy for ambient noise level."""
        return float(self.last_rms)

    def close(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
