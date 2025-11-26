# SentiGuard Cloud Deployment Guide

## Deploying to Railway (Recommended) 🚂

Railway offers the simplest deployment with automatic PostgreSQL provisioning.

### Step 1: Install Railway CLI

```bash
npm install -g @railway/cli
```

### Step 2: Login to Railway

```bash
railway login
```

### Step 3: Initialize Project

```bash
cd sra/backend
railway init
```

### Step 4: Add PostgreSQL Database

In the Railway dashboard:
1. Click "New" → "Database" → "PostgreSQL"
2. Railway will automatically set `DATABASE_URL` environment variable

### Step 5: Configure Environment Variables

In Railway dashboard, go to your service → Variables:

```
TWILIO_ACCOUNT_SID=your_sid_here
TWILIO_AUTH_TOKEN=your_token_here
TWILIO_FROM_NUMBER=+1234567890
TWILIO_TO_NUMBERS=+0987654321
DEBUG=False
ALLOWED_ORIGINS=https://your-app.railway.app
```

### Step 6: Deploy

```bash
railway up
```

Your app will be live at: `https://your-app.railway.app`

Access dashboard: `https://your-app.railway.app/dashboard.html`

---

## Deploying to Render 🎨

### Step 1: Connect Repository

1. Go to [render.com](https://render.com)
2. Click "New +" → "Web Service"
3. Connect your Git repository

### Step 2: Configure Service

- **Name**: sentiguard
- **Environment**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python main.py`
- **Plan**: Free or Starter

### Step 3: Add PostgreSQL

1. Click "New +" → "PostgreSQL"
2. Create database
3. Copy "Internal Database URL"

### Step 4: Environment Variables

In your web service settings → Environment:

```
DATABASE_URL=<internal-database-url>
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_FROM_NUMBER=+1234567890
TWILIO_TO_NUMBERS=+0987654321
DEBUG=False
```

### Step 5: Deploy

Click "Manual Deploy" → "Deploy latest commit"

Your app will be live at: `https://sentiguard.onrender.com`

---

## Deploying to Fly.io ✈️

Fly.io offers edge deployment for low-latency globally.

### Step 1: Install Fly CLI

```bash
# macOS/Linux
curl -L https://fly.io/install.sh | sh

# Windows
powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
```

### Step 2: Login

```bash
fly auth login
```

### Step 3: Launch App

```bash
cd sra/backend
fly launch
```

Answer prompts:
- App name: sentiguard (or your choice)
- Region: Choose closest to you
- PostgreSQL: Yes
- Deploy now: No (we'll set env vars first)

### Step 4: Attach PostgreSQL

```bash
fly postgres create
fly postgres attach <postgres-app-name>
```

### Step 5: Set Environment Variables

```bash
fly secrets set TWILIO_ACCOUNT_SID=your_sid
fly secrets set TWILIO_AUTH_TOKEN=your_token
fly secrets set TWILIO_FROM_NUMBER=+1234567890
fly secrets set TWILIO_TO_NUMBERS=+0987654321
fly secrets set DEBUG=False
```

### Step 6: Deploy

```bash
fly deploy
```

Your app: `https://sentiguard.fly.dev`

---

## Post-Deployment Checklist ✅

After deploying to any platform:

1. **Test Health Endpoint**
   ```bash
   curl https://your-app-url/health
   ```

2. **Access Dashboard**
   ```
   https://your-app-url/dashboard.html
   ```

3. **Test Camera/Upload**
   - Try switching to sample video
   - Upload a test video

4. **Test Alarm System**
   ```bash
   curl -X POST https://your-app-url/alarm/test
   ```

5. **Monitor Logs**
   - Railway: `railway logs`
   - Render: View in dashboard
   - Fly.io: `fly logs`

6. **Set Up Custom Domain** (Optional)
   - Railway: Settings → Domains
   - Render: Settings → Custom Domain
   - Fly.io: `fly certs add yourdomain.com`

---

## Troubleshooting 🔧

### Database Connection Issues

```bash
# Check DATABASE_URL format
postgresql+asyncpg://user:pass@host:port/dbname

# Test connection
python -c "import asyncpg; asyncpg.connect('your-url')"
```

### WebSocket Not Connecting

- Ensure `ALLOWED_ORIGINS` includes your domain
- Check if platform supports WebSockets (all three do)

### Alarm Not Triggering

- Verify Twilio credentials in environment variables
- Check alarm threshold (default: 60%)

### High CPU Usage

- Reduce FPS from 60 to 30 in `main.py` (line 213)
- Enable frame skipping for cloud deployment

---

## Scaling Considerations 📈

### Horizontal Scaling

All platforms support:
- Multiple instances (load balancing)
- Auto-scaling based on traffic

### Database Optimization

```sql
-- Add indexes for common queries
CREATE INDEX idx_alarm_timestamp ON alarm_events(timestamp DESC);
CREATE INDEX idx_sensor_name_timestamp ON sensor_readings(sensor_name, timestamp DESC);
```

### Caching

Add Redis for sensor data caching:

```yaml
# docker-compose.yml
redis:
  image: redis:alpine
  ports:
    - "6379:6379"
```

---

## Cost Estimates 💰

### Railway
- Free tier: $5/month credit
- Starter: $5/month per service
- PostgreSQL: Included

### Render
- Free tier: Available (with limitations)
- Starter: $7/month
- PostgreSQL: $7/month

### Fly.io
- Free tier: 3 VMs, 3GB storage
- Paid: ~$5-20/month depending on usage

**Recommended for production**: Railway Starter ($10-15/month total)

---

## Support 📧

Need help deploying? Open an issue on GitHub with:
- Platform name
- Error logs
- Environment configuration (sanitized)
