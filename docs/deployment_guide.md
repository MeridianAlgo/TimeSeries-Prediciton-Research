# Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Enhanced Time Series Prediction System in various environments, from local development to production cloud deployments.

## Table of Contents

1. [Local Development Setup](#local-setup)
2. [Docker Deployment](#docker)
3. [Cloud Deployment](#cloud)
4. [Production Setup](#production)
5. [Monitoring and Maintenance](#monitoring)
6. [Performance Tuning](#performance)
7. [Troubleshooting](#troubleshooting)

## Local Development Setup <a name="local-setup"></a>

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for acceleration)
- 8GB+ RAM
- 10GB+ free disk space

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MeridianLearning
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyTorch with CUDA support (optional)**
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CPU only
   pip install torch torchvision torchaudio
   ```

5. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

### Configuration

1. **Create configuration file**
   ```bash
   cp config/config.yaml.example config/config.yaml
   ```

2. **Edit configuration**
   ```yaml
   # config/config.yaml
   models:
     transformer:
       input_dim: 100
       d_model: 256
       nhead: 16
       num_layers: 8
       seq_len: 60
       dropout: 0.1
     
     lstm:
       input_dim: 100
       hidden_dim: 256
       num_layers: 4
       bidirectional: true
       attention: true
       dropout: 0.2
     
     cnn_lstm:
       input_dim: 100
       cnn_channels: [32, 64, 128]
       lstm_hidden: 256
       seq_len: 60
       dropout: 0.2

   training:
     batch_size: 32
     learning_rate: 0.001
     epochs: 100
     early_stopping_patience: 10
     validation_split: 0.2

   ensemble:
     weighting_method: "performance_based"
     uncertainty_method: "ensemble_variance"
     performance_window: 30

   monitoring:
     accuracy_threshold: 0.7
     alert_window_minutes: 30
     performance_window_days: 30
     log_level: "INFO"

   data:
     sequence_length: 60
     feature_columns: ["open", "high", "low", "close", "volume"]
     target_column: "close"
     test_size: 0.2
   ```

### Running the System

1. **Basic usage**
   ```bash
   python main.py --config config/config.yaml
   ```

2. **Training only**
   ```bash
   python main.py --mode train --config config/config.yaml
   ```

3. **Prediction only**
   ```bash
   python main.py --mode predict --config config/config.yaml
   ```

4. **Backtesting**
   ```bash
   python main.py --mode backtest --config config/config.yaml
   ```

## Docker Deployment <a name="docker"></a>

### Dockerfile

```dockerfile
# Use CUDA base image for GPU support
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs models data config

# Expose port for web interface
EXPOSE 8080

# Set default command
CMD ["python3", "main.py", "--config", "config/config.yaml"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  timeseries-predictor:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
      - ./config:/app/config
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - PYTHONPATH=/app
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: timeseries
      POSTGRES_USER: predictor
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
```

### Building and Running

1. **Build the image**
   ```bash
   docker build -t timeseries-predictor .
   ```

2. **Run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

3. **Check logs**
   ```bash
   docker-compose logs -f timeseries-predictor
   ```

4. **Stop services**
   ```bash
   docker-compose down
   ```

## Cloud Deployment <a name="cloud"></a>

### AWS Deployment

#### EC2 Setup

1. **Launch EC2 instance**
   ```bash
   # Use Deep Learning AMI with CUDA support
   aws ec2 run-instances \
     --image-id ami-0c02fb55956c7d316 \
     --instance-type g4dn.xlarge \
     --key-name your-key-pair \
     --security-group-ids sg-xxxxxxxxx \
     --subnet-id subnet-xxxxxxxxx
   ```

2. **Install dependencies**
   ```bash
   # Connect to instance
   ssh -i your-key.pem ubuntu@your-instance-ip

   # Update system
   sudo apt-get update && sudo apt-get upgrade -y

   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER

   # Install Docker Compose
   sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

3. **Deploy application**
   ```bash
   # Clone repository
   git clone <repository-url>
   cd MeridianLearning

   # Build and run
   docker-compose up -d
   ```

#### ECS Deployment

1. **Create ECR repository**
   ```bash
   aws ecr create-repository --repository-name timeseries-predictor
   ```

2. **Build and push image**
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
   docker build -t timeseries-predictor .
   docker tag timeseries-predictor:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/timeseries-predictor:latest
   docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/timeseries-predictor:latest
   ```

3. **Create ECS task definition**
   ```json
   {
     "family": "timeseries-predictor",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "2048",
     "memory": "4096",
     "executionRoleArn": "arn:aws:iam::<account-id>:role/ecsTaskExecutionRole",
     "containerDefinitions": [
       {
         "name": "timeseries-predictor",
         "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/timeseries-predictor:latest",
         "portMappings": [
           {
             "containerPort": 8080,
             "protocol": "tcp"
           }
         ],
         "environment": [
           {
             "name": "PYTHONPATH",
             "value": "/app"
           }
         ],
         "logConfiguration": {
           "logDriver": "awslogs",
           "options": {
             "awslogs-group": "/ecs/timeseries-predictor",
             "awslogs-region": "us-east-1",
             "awslogs-stream-prefix": "ecs"
           }
         }
       }
     ]
   }
   ```

### Google Cloud Platform

#### GKE Deployment

1. **Create GKE cluster**
   ```bash
   gcloud container clusters create timeseries-cluster \
     --zone us-central1-a \
     --num-nodes 3 \
     --machine-type n1-standard-4 \
     --enable-autoscaling \
     --min-nodes 1 \
     --max-nodes 10
   ```

2. **Deploy to GKE**
   ```bash
   # Build and push to Container Registry
   docker build -t gcr.io/<project-id>/timeseries-predictor .
   docker push gcr.io/<project-id>/timeseries-predictor

   # Deploy to GKE
   kubectl apply -f k8s/
   ```

#### Kubernetes Manifests

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: timeseries-predictor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: timeseries-predictor
  template:
    metadata:
      labels:
        app: timeseries-predictor
    spec:
      containers:
      - name: timeseries-predictor
        image: gcr.io/<project-id>/timeseries-predictor:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: PYTHONPATH
          value: "/app"
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: data-volume
          mountPath: /app/data
      volumes:
      - name: config-volume
        configMap:
          name: timeseries-config
      - name: data-volume
        persistentVolumeClaim:
          claimName: timeseries-data-pvc
```

### Azure Deployment

#### AKS Setup

1. **Create AKS cluster**
   ```bash
   az aks create \
     --resource-group myResourceGroup \
     --name timeseries-cluster \
     --node-count 3 \
     --enable-addons monitoring \
     --generate-ssh-keys
   ```

2. **Deploy application**
   ```bash
   # Build and push to Azure Container Registry
   az acr build --registry myacr --image timeseries-predictor .

   # Deploy to AKS
   kubectl apply -f k8s/
   ```

## Production Setup <a name="production"></a>

### High Availability Configuration

1. **Load Balancer Setup**
   ```yaml
   # k8s/service.yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: timeseries-predictor-service
   spec:
     type: LoadBalancer
     ports:
     - port: 80
       targetPort: 8080
     selector:
       app: timeseries-predictor
   ```

2. **Auto-scaling**
   ```yaml
   # k8s/hpa.yaml
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: timeseries-predictor-hpa
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: timeseries-predictor
     minReplicas: 3
     maxReplicas: 10
     metrics:
     - type: Resource
       resource:
         name: cpu
         target:
           type: Utilization
           averageUtilization: 70
     - type: Resource
       resource:
         name: memory
         target:
           type: Utilization
           averageUtilization: 80
   ```

### Database Setup

1. **PostgreSQL Configuration**
   ```sql
   -- Create database and user
   CREATE DATABASE timeseries_prediction;
   CREATE USER predictor_user WITH PASSWORD 'secure_password';
   GRANT ALL PRIVILEGES ON DATABASE timeseries_prediction TO predictor_user;

   -- Create tables
   CREATE TABLE predictions (
       id SERIAL PRIMARY KEY,
       timestamp TIMESTAMP NOT NULL,
       asset_symbol VARCHAR(10) NOT NULL,
       predicted_value DECIMAL(10,4) NOT NULL,
       actual_value DECIMAL(10,4),
       confidence_score DECIMAL(5,4),
       model_name VARCHAR(50),
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );

   CREATE TABLE model_performance (
       id SERIAL PRIMARY KEY,
       model_name VARCHAR(50) NOT NULL,
       metric_name VARCHAR(20) NOT NULL,
       metric_value DECIMAL(10,6) NOT NULL,
       timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );

   CREATE INDEX idx_predictions_timestamp ON predictions(timestamp);
   CREATE INDEX idx_predictions_asset ON predictions(asset_symbol);
   ```

2. **Redis Configuration**
   ```bash
   # redis.conf
   maxmemory 2gb
   maxmemory-policy allkeys-lru
   save 900 1
   save 300 10
   save 60 10000
   ```

### Security Configuration

1. **SSL/TLS Setup**
   ```yaml
   # k8s/ingress.yaml
   apiVersion: networking.k8s.io/v1
   kind: Ingress
   metadata:
     name: timeseries-predictor-ingress
     annotations:
       kubernetes.io/ingress.class: nginx
       cert-manager.io/cluster-issuer: letsencrypt-prod
   spec:
     tls:
     - hosts:
       - predictor.yourdomain.com
       secretName: timeseries-predictor-tls
     rules:
     - host: predictor.yourdomain.com
       http:
         paths:
         - path: /
           pathType: Prefix
           backend:
             service:
               name: timeseries-predictor-service
               port:
                 number: 80
   ```

2. **Secrets Management**
   ```bash
   # Create Kubernetes secrets
   kubectl create secret generic timeseries-secrets \
     --from-literal=db-password=secure_password \
     --from-literal=api-key=your_api_key \
     --from-literal=redis-password=redis_password
   ```

## Monitoring and Maintenance <a name="monitoring"></a>

### Monitoring Stack

1. **Prometheus Configuration**
   ```yaml
   # prometheus.yml
   global:
     scrape_interval: 15s
   
   scrape_configs:
     - job_name: 'timeseries-predictor'
       static_configs:
         - targets: ['timeseries-predictor-service:8080']
       metrics_path: '/metrics'
   ```

2. **Grafana Dashboards**
   ```json
   {
     "dashboard": {
       "title": "Time Series Predictor Metrics",
       "panels": [
         {
           "title": "Prediction Accuracy",
           "type": "graph",
           "targets": [
             {
               "expr": "prediction_accuracy",
               "legendFormat": "Accuracy"
             }
           ]
         },
         {
           "title": "Model Performance",
           "type": "graph",
           "targets": [
             {
               "expr": "model_inference_time",
               "legendFormat": "Inference Time"
             }
           ]
         }
       ]
     }
   }
   ```

### Logging Configuration

1. **Structured Logging**
   ```python
   import logging
   import json
   from datetime import datetime

   class StructuredFormatter(logging.Formatter):
       def format(self, record):
           log_entry = {
               'timestamp': datetime.utcnow().isoformat(),
               'level': record.levelname,
               'logger': record.name,
               'message': record.getMessage(),
               'module': record.module,
               'function': record.funcName,
               'line': record.lineno
           }
           
           if hasattr(record, 'extra_fields'):
               log_entry.update(record.extra_fields)
           
           return json.dumps(log_entry)

   # Configure logging
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       handlers=[
           logging.FileHandler('logs/app.log'),
           logging.StreamHandler()
       ]
   )
   ```

2. **Log Rotation**
   ```python
   from logging.handlers import RotatingFileHandler

   handler = RotatingFileHandler(
       'logs/app.log',
       maxBytes=10*1024*1024,  # 10MB
       backupCount=5
   )
   ```

### Health Checks

1. **Application Health Endpoint**
   ```python
   from flask import Flask, jsonify
   import psutil
   import torch

   app = Flask(__name__)

   @app.route('/health')
   def health_check():
       return jsonify({
           'status': 'healthy',
           'timestamp': datetime.utcnow().isoformat(),
           'system': {
               'cpu_percent': psutil.cpu_percent(),
               'memory_percent': psutil.virtual_memory().percent,
               'disk_percent': psutil.disk_usage('/').percent
           },
           'gpu': {
               'available': torch.cuda.is_available(),
               'memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
           },
           'models': {
               'transformer': model_status['transformer'],
               'lstm': model_status['lstm'],
               'cnn_lstm': model_status['cnn_lstm']
           }
       })

   @app.route('/ready')
   def readiness_check():
       # Check if models are loaded and ready
       if all(model_status.values()):
           return jsonify({'status': 'ready'}), 200
       else:
           return jsonify({'status': 'not ready'}), 503
   ```

2. **Kubernetes Health Checks**
   ```yaml
   # In deployment.yaml
   livenessProbe:
     httpGet:
       path: /health
       port: 8080
     initialDelaySeconds: 30
     periodSeconds: 10
   readinessProbe:
     httpGet:
       path: /ready
       port: 8080
     initialDelaySeconds: 5
     periodSeconds: 5
   ```

## Performance Tuning <a name="performance"></a>

### Model Optimization

1. **Mixed Precision Training**
   ```python
   from torch.cuda.amp import autocast, GradScaler

   scaler = GradScaler()

   for epoch in range(epochs):
       for batch in dataloader:
           optimizer.zero_grad()
           
           with autocast():
               outputs = model(batch)
               loss = criterion(outputs, targets)
           
           scaler.scale(loss).backward()
           scaler.step(optimizer)
           scaler.update()
   ```

2. **Model Quantization**
   ```python
   import torch.quantization as quantization

   # Quantize model for inference
   quantized_model = quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   ```

3. **Batch Processing Optimization**
   ```python
   # Optimize batch size based on GPU memory
   def get_optimal_batch_size(model, input_shape, max_memory_gb=8):
       device = next(model.parameters()).device
       max_memory = max_memory_gb * 1024 * 1024 * 1024  # Convert to bytes
       
       batch_size = 1
       while True:
           try:
               test_input = torch.randn(batch_size, *input_shape).to(device)
               with torch.no_grad():
                   _ = model(test_input)
               batch_size *= 2
           except RuntimeError:
               return batch_size // 2
   ```

### Database Optimization

1. **Connection Pooling**
   ```python
   import psycopg2
   from psycopg2 import pool

   connection_pool = psycopg2.pool.SimpleConnectionPool(
       1, 20,  # minconn, maxconn
       host="localhost",
       database="timeseries_prediction",
       user="predictor_user",
       password="secure_password"
   )
   ```

2. **Query Optimization**
   ```sql
   -- Create indexes for common queries
   CREATE INDEX CONCURRENTLY idx_predictions_model_timestamp 
   ON predictions(model_name, timestamp);

   CREATE INDEX CONCURRENTLY idx_performance_model_metric 
   ON model_performance(model_name, metric_name);

   -- Partition tables by date
   CREATE TABLE predictions_2024 PARTITION OF predictions
   FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
   ```

### Caching Strategy

1. **Redis Caching**
   ```python
   import redis
   import pickle

   redis_client = redis.Redis(host='localhost', port=6379, db=0)

   def cache_prediction(key, prediction, ttl=3600):
       redis_client.setex(key, ttl, pickle.dumps(prediction))

   def get_cached_prediction(key):
       data = redis_client.get(key)
       return pickle.loads(data) if data else None
   ```

2. **Model Caching**
   ```python
   class ModelCache:
       def __init__(self, max_size=10):
           self.cache = {}
           self.max_size = max_size
       
       def get_model(self, model_name, config):
           key = f"{model_name}_{hash(str(config))}"
           if key in self.cache:
               return self.cache[key]
           
           if len(self.cache) >= self.max_size:
               # Remove least recently used
               oldest_key = next(iter(self.cache))
               del self.cache[oldest_key]
           
           model = self.load_model(model_name, config)
           self.cache[key] = model
           return model
   ```

## Troubleshooting <a name="troubleshooting"></a>

### Common Issues

1. **Out of Memory Errors**
   ```bash
   # Check GPU memory usage
   nvidia-smi
   
   # Reduce batch size in config
   training:
     batch_size: 16  # Reduce from 32
   
   # Enable gradient accumulation
   training:
     gradient_accumulation_steps: 2
   ```

2. **Slow Training**
   ```bash
   # Check GPU utilization
   watch -n 1 nvidia-smi
   
   # Profile training
   python -m torch.utils.bottleneck main.py
   
   # Use mixed precision
   training:
     use_amp: true
   ```

3. **Model Convergence Issues**
   ```python
   # Adjust learning rate
   training:
     learning_rate: 0.0001  # Reduce from 0.001
   
   # Use learning rate scheduler
   scheduler:
     type: "cosine"
     warmup_steps: 1000
   ```

### Debug Mode

1. **Enable Debug Logging**
   ```yaml
   # config/config.yaml
   logging:
     level: "DEBUG"
     handlers:
       - type: "file"
         filename: "logs/debug.log"
       - type: "console"
   ```

2. **Model Debugging**
   ```python
   # Enable PyTorch debug mode
   torch.autograd.set_detect_anomaly(True)
   
   # Check for NaN values
   def check_nan(tensor, name):
       if torch.isnan(tensor).any():
           print(f"NaN detected in {name}")
           return True
       return False
   ```

### Performance Monitoring

1. **Resource Monitoring**
   ```bash
   # Monitor CPU and memory
   htop
   
   # Monitor GPU
   nvidia-smi -l 1
   
   # Monitor disk I/O
   iostat -x 1
   ```

2. **Application Metrics**
   ```python
   # Custom metrics
   from prometheus_client import Counter, Histogram, Gauge
   
   prediction_counter = Counter('predictions_total', 'Total predictions made')
   inference_time = Histogram('inference_duration_seconds', 'Time spent on inference')
   model_accuracy = Gauge('model_accuracy', 'Current model accuracy')
   ```

### Backup and Recovery

1. **Model Backup**
   ```bash
   # Backup trained models
   tar -czf models_backup_$(date +%Y%m%d).tar.gz models/
   
   # Upload to cloud storage
   aws s3 cp models_backup_$(date +%Y%m%d).tar.gz s3://your-bucket/backups/
   ```

2. **Database Backup**
   ```bash
   # PostgreSQL backup
   pg_dump timeseries_prediction > backup_$(date +%Y%m%d).sql
   
   # Automated backup script
   #!/bin/bash
   DATE=$(date +%Y%m%d_%H%M%S)
   pg_dump timeseries_prediction | gzip > backup_$DATE.sql.gz
   aws s3 cp backup_$DATE.sql.gz s3://your-bucket/db-backups/
   ```

### Support and Maintenance

1. **Regular Maintenance Tasks**
   ```bash
   # Weekly tasks
   - Clean old log files
   - Update model performance metrics
   - Check disk space usage
   - Review error logs
   
   # Monthly tasks
   - Update dependencies
   - Retrain models with new data
   - Review and optimize queries
   - Update security patches
   ```

2. **Emergency Procedures**
   ```bash
   # Rollback to previous version
   kubectl rollout undo deployment/timeseries-predictor
   
   # Restart services
   docker-compose restart
   
   # Scale down during issues
   kubectl scale deployment timeseries-predictor --replicas=1
   ```

This deployment guide provides comprehensive instructions for deploying the Enhanced Time Series Prediction System in various environments. Follow the sections relevant to your deployment scenario and ensure proper testing before moving to production.
