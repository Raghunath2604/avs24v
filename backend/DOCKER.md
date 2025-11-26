# Docker Quick Start Guide

## Running SentiGuard with Docker

### Prerequisites
- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose installed (included with Docker Desktop)

### One-Command Start 🚀

```bash
# Navigate to backend directory
cd sra/backend

# Start everything
docker-compose up -d
```

That's it! SentiGuard is now running with:
- ✅ Backend API at `http://localhost:8001`
- ✅ PostgreSQL database
- ✅ Dashboard at `http://localhost:8001/dashboard.html`

### Useful Commands

```bash
# View logs
docker-compose logs -f

# View only backend logs
docker-compose logs -f backend

# Restart services
docker-compose restart

# Stop everything
docker-compose down

# Stop and remove volumes (fresh start)
docker-compose down -v

# Rebuild after code changes
docker-compose up -d --build
```

### Accessing the Application

- **Dashboard**: http://localhost:8001/dashboard.html
- **API Docs**: http://localhost:8001/docs
- **Health Check**: http://localhost:8001/health

### Environment Variables

Edit `docker-compose.yml` to configure:

```yaml
environment:
  - TWILIO_ACCOUNT_SID=your_sid_here
  - TWILIO_AUTH_TOKEN=your_token_here
  - DEBUG=True
```

Or create `.env` file:

```bash
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_FROM_NUMBER=+1234567890
TWILIO_TO_NUMBERS=+0987654321
```

### Troubleshooting

**Port already in use?**
```bash
# Change port in docker-compose.yml
ports:
  - "8002:8001"  # Use 8002 instead
```

**Database connection issues?**
```bash
# Check if postgres is healthy
docker-compose ps

# View postgres logs
docker-compose logs postgres
```

**Need fresh start?**
```bash
docker-compose down -v  # Remove all data
docker-compose up -d    # Start fresh
```

### Production Deployment

For production, use:

```bash
docker-compose -f docker-compose.prod.yml up -d
```

Or deploy to cloud using our deployment guides:
- See [DEPLOYMENT.md](DEPLOYMENT.md) for Railway, Render, Fly.io
