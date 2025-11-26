# SentiGuard 🛡️
## Enterprise-Grade AI Security Monitoring System

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue.svg)](https://www.postgresql.org/)

Full-stack security monitoring with **60 FPS** streaming, **50 AI sensors**, real-time alarms, and SMS notifications.

## 🚀 Features

- **60 FPS Real-time Streaming** - Smooth, low-latency video processing
- **50 AI Sensors** - Vision, audio, behavior, and system monitoring
- **Enhanced Weapon Detection** - 2x priority for firearms, knives, explosives
- **Intelligent Alarm System** - Auto-trigger at 60% threat with 5s alarm
- **SMS Notifications** - Twilio integration for critical alerts
- **PostgreSQL Database** - Persistent storage for events & analytics
- **Docker Ready** - One-command deployment
- **Cloud Deployable** - Railway, Render, Fly.io configs included

## 📋 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (for containerized deployment)
- PostgreSQL 15+ (or use Docker Compose)

### Local Development

```bash
# Clone repository
git clone <your-repo-url>
cd sra/backend

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your configuration

# Run locally
python main.py
```

Access dashboard: `http://localhost:8001/dashboard.html`

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop
docker-compose down
```

## 🏗️ Architecture

```
┌─────────────────┐
│  Dashboard.html │ ← Frontend (HTML/CSS/JS)
└────────┬────────┘
         │ WebSocket (60 FPS)
         ↓
┌─────────────────┐
│   FastAPI App   │ ← Backend (Python)
└────────┬────────┘
         │
    ┌────┴────┐
    ↓         ↓
┌──────┐  ┌──────────┐
│Camera│  │PostgreSQL│
└──────┘  └──────────┘
```

## 📁 Project Structure

```
sra/backend/
├── main.py              # FastAPI application
├── database.py          # Database configuration
├── models.py            # SQLAlchemy models
├── alarm_system.py      # Alarm logic & SMS
├── vision.py            # AI vision detection
├── audio.py             # Audio processing
├── virtual_sensors.py   # 50 sensor implementations
├── fusion.py            # Sensor fusion engine
├── dashboard.html       # Frontend UI
├── Dockerfile           # Docker build config
├── docker-compose.yml   # Multi-service orchestration
├── requirements.txt     # Python dependencies
└── .env.example         # Environment template
```

## 🌐 Cloud Deployment

### Railway (Recommended)

1. Install Railway CLI:
   ```bash
   npm install -g @railway/cli
   ```

2. Login and deploy:
   ```bash
   railway login
   railway init
   railway up
   ```

3. Add PostgreSQL service in Railway dashboard
4. Set environment variables in Railway settings

### Render

1. Connect your Git repository
2. Create new Web Service
3. Build command: `pip install -r requirements.txt`
4. Start command: `python main.py`
5. Add PostgreSQL database
6. Configure environment variables

### Fly.io

```bash
fly launch
fly postgres create
fly postgres attach
fly deploy
```

## 🔧 Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/sentiguard

# Twilio SMS (Optional)
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_FROM_NUMBER=+1234567890
TWILIO_TO_NUMBERS=+0987654321

# App Config
DEBUG=False
ALLOWED_ORIGINS=https://yourdomain.com
```

## 📊 Database Models

- **AlarmEvent** - Historical alarm records
- **SensorReading** - Time-series sensor data
- **VideoUpload** - Uploaded video metadata
- **SystemLog** - Application logs

## 🎯 API Endpoints

### REST API
- `POST /upload/video` - Upload video file
- `POST /config/source` - Switch video source (camera/sample/upload)
- `POST /alarm/test` - Test alarm system
- `GET /alarm/history` - Get alarm history
- `GET /health` - Health check

### WebSocket
- `WS /ws` - Real-time video & sensor streaming

## 🛠️ Development

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn main:app --reload --port 8001

# Database migrations
alembic revision --autogenerate -m "description"
alembic upgrade head
```

## 📈 Performance

- **60 FPS** streaming
- **<100ms** latency
- **50 sensors** processing in real-time
- Smart inference (every 2nd frame) for efficiency

## 🔒 Security Features

- Enhanced weapon detection (0.30 priority weight)
- Multi-sensor fusion for accurate threat assessment
- Configurable alarm thresholds
- SMS notifications for critical events
- Event logging and audit trail

## 📝 License

[Your License Here]

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## 📧 Support

For issues or questions, please open a GitHub issue.

---

**Built with ❤️ for next-generation security monitoring**
