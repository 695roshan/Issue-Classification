# Issue Classification Application

A web application that automatically classifies Jira-like issues into categories (bug, enhancement, question) using Word2Vec embeddings and Random Forest classification.

## Features

- **Automated Issue Classification**: Classify issues as bug, enhancement, or question
- **Option to correct wrongly predicted category**: If the user thinks the prediction is incorrect, it can be changed.

- **Multi-language Support**: Detects and validates issue language (English only)
- **Supabase Integration**: Persistent storage and retrieval of classifications
- **Prometheus Metrics**: Real-time monitoring of predictions and accuracy
- **Comprehensive Testing**: Unit tests with coverage reporting

## Prerequisites

- Python 3.12.3 or higher
- Docker and Docker Compose (for containerized setup)
- Supabase account (for database integration)

## Setup Instructions

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/695roshan/Issue-Classification.git
   cd Issue-Classification
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create and configure the .env file** 
    Create a `.env` file in the root directory with the following variables:

    ```bash
    # Supabase Configuration
    SUPABASE_URL=https://your-project-id.supabase.co
    SUPABASE_KEY=your-supabase-anon-key
    ```

5. **Run the application**
   ```bash
   python app.py
   ```
   **The application will be available at**: `http://localhost:5000`

### Docker Setup

Docker is used to run Prometheus and Grafana for monitoring, while the Flask server runs locally from the command line.

**Start Docker services:**

1. **Ensure the .env file is created** with your Supabase credentials

2. **Start Docker Compose** (Prometheus & Grafana)
   ```bash
   docker-compose up -d
   ```

   This will start:
   - **Prometheus**: http://localhost:9090
   - **Grafana**: http://localhost:3000

3. **Stop Docker services**
   ```bash
   docker-compose down
   ```

## Monitoring with Prometheus and Grafana

### Prometheus Setup

Prometheus runs in Docker and automatically collects metrics from your Flask application at the `/metrics` endpoint. The `prometheus.yml` configuration file specifies the Flask app target.

**Note:** When Flask runs locally (not in Docker) and Prometheus runs in Docker, Prometheus accesses Flask via `host.docker.internal:5000` (on Windows) or your machine's IP address (on Linux/Mac).

**Key metrics monitored:**
- `accuracy` - Model accuracy
- `avg_prediction_confidence` - Average confidence of predictions
- `predictions_per_category` - Number of predictions per label
- `correct_predictions_per_category` - Correct predictions per label
- `incorrect_predictions_per_category` - Incorrect predictions per label
- `request_latency_seconds` - API request latency

**Access Prometheus:**
- URL: `http://localhost:9090`
- Query metrics using PromQL in the Query interface
- Example query: `accuracy`

### Grafana Setup

Grafana provides visualization dashboards for your metrics.

**First-time setup:**

1. Access Grafana at `http://localhost:3000`
2. Login with default credentials:
   - Username: `admin`
   - Password: `admin`
3. Add Prometheus as a data source:
   - Click **Configuration** → **Data Sources**
   - Click **Add data source**
   - Select **Prometheus**
   - Set URL to `http://prometheus:9090`
   - Click **Save & Test**
4. Create a dashboard:
   - Click **+** → **Dashboard**
   - Click **Add new panel**
   - Select **Prometheus** as data source
   - Add queries for metrics you want to visualize
   - Example queries:
     - `accuracy` - Display model accuracy
     - `avg_prediction_confidence` - Display average confidence
     - `rate(predictions_per_category[5m])` - Show prediction rate

**Useful Grafana Features:**
- Set refresh intervals for real-time updates
- Create alerts based on metric thresholds
- Use templating for dynamic dashboards
- Export and share dashboards

## Running Tests

### Local Testing

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=app --cov=model --cov=ticket --cov-report=term-missing

# Run specific test file
pytest tests/test_app.py

# Run with verbose output
pytest -v
```
