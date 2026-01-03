
# backend/config.py

# -----------------------------------------------------
# HAWKEYE CONFIGURATION
# Central place to switch between MOCK data and REAL SENSORS
# -----------------------------------------------------

# MASTER SWITCHES
# Set to False to try and connect to the real endpoints below.
USE_MOCK_VISION = False
USE_MOCK_AUDIO = False
USE_MOCK_MOTION = False

# MEMBER ENDPOINTS (If running on localhost ports)
# Vision (Member 1)
VISION_API_URL = "http://127.0.0.1:8001" 

# Audio (Member 2)
AUDIO_API_URL = "http://127.0.0.1:8002"

# Motion (Member 3)
MOTION_API_URL = "http://127.0.0.1:8003"
