# 🔥 Firestorm Innovators – Earth System Trend Detective

### Backend | NASA Space Apps Challenge 2026

**Turning Earth observation data into meaningful environmental insights.**

Firestorm is the backend system developed for NASA Space Apps Challenge 2026 – *Earth System Trend Detective*. It focuses on collecting, processing, analyzing, and managing environmental and climate-related datasets to identify meaningful patterns and long-term changes in Earth's systems.

The backend integrates NASA POWER data services with scientific data-processing libraries to support environmental trend analysis and data-driven insights.

---

## 🌍 Project Overview

Understanding how Earth's environmental systems change over time is essential for identifying climate patterns and supporting informed decisions.

Firestorm aims to simplify this process through an automated backend pipeline that works with environmental datasets, processes historical observations, and prepares analytical results for visualization and interpretation.

The system is designed around:

- Environmental data acquisition.
- Historical climate data processing.
- Time-series analysis.
- Statistical trend detection.
- Data storage and management.
- API-based communication with the frontend.

---

## 🎯 Objectives

1. Retrieve environmental and climate-related data using NASA POWER.
2. Process and organize historical environmental datasets.
3. Identify patterns and trends in environmental variables.
4. Apply statistical methods to analyze changes over time.
5. Provide structured data through REST APIs.
6. Maintain a scalable backend architecture for future expansion.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core backend development |
| FastAPI | REST API development |
| Uvicorn | ASGI application server |
| NASA POWER API | Environmental data source |
| Pandas | Data manipulation and processing |
| NumPy | Numerical computations |
| SciPy | Statistical analysis |
| SQLAlchemy | Database operations |
| SQLite | Local database |
| PostgreSQL | Production database support |
| Alembic | Database migrations |
| HTTPX | HTTP requests |
| Pydantic | Data validation |
| Pytest | Backend testing |

---

## 🏗️ Backend Architecture

The backend follows a modular architecture that separates application responsibilities.

```text
firestorm-backend/
│
├── app/
│   ├── API / Routes
│   ├── Services
│   ├── Models
│   ├── Database
│   └── Configuration
│
├── scripts/
│   └── Utility and execution scripts
│
├── tests/
│   └── Backend test cases
│
├── .env.example
├── .gitignore
├── power_probe.json
├── requirements.txt
└── README.md
```

*Note: The directory descriptions above are conceptual; refer to the implementation for the exact module names and responsibilities.*

---

## 🛰️ Data Source – NASA POWER

The project uses the **NASA Prediction Of Worldwide Energy Resources (POWER)** service.

NASA POWER provides environmental and meteorological datasets that can be used for climate-related research and analysis.

Official resource:

https://power.larc.nasa.gov/

The backend is configured to communicate with NASA POWER through its API.

### Data Processing Pipeline

```text
       NASA POWER API
              |
              v
      Data Acquisition
              |
              v
       Data Validation
              |
              v
      Data Preprocessing
              |
              v
      Statistical Analysis
              |
              v
       Trend Detection
              |
              v
       Database Storage
              |
              v
        REST API Output
              |
              v
       Frontend Dashboard
```

---

## 📊 Environmental Data Analysis

The backend uses scientific Python libraries to support environmental data analysis.

Core analytical capabilities are intended to include:

- Time-series data processing.
- Historical data comparison.
- Statistical calculations.
- Environmental trend identification.
- Numerical data analysis.
- Preparation of structured analytical results.

---

## ⚙️ Installation & Setup

Follow these instructions to run the backend locally.

### 1. Clone the Repository

```bash
git clone https://github.com/vanshmodanwal01/firestorm-backend.git

cd firestorm-backend
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create your local `.env` file using the example configuration.

```bash
copy .env.example .env
```

For Linux/macOS:

```bash
cp .env.example .env
```

Configure the following variables:

```env
DATABASE_URL=sqlite:///./nasa_trends.db

LOG_LEVEL=INFO

POWER_BASE_URL=https://power.larc.nasa.gov

POWER_MAX_REQUESTS_PER_MINUTE=30

POWER_TIMEOUT_SECONDS=60

POWER_MAX_RETRIES=3

CACHE_TTL_HOURS=168

CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

**Important:** NASA POWER data access does not require an API key.

Never upload your actual `.env` file or private credentials to GitHub.

### 5. Run the Backend

Start the FastAPI application using the project's configured ASGI application entry point.

```bash
uvicorn <application_module>:app --reload
```

Replace `<application_module>` with the actual module containing your FastAPI application instance.

### 6. API Documentation

Once the backend is running, FastAPI provides interactive API documentation.

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

---

## 🧪 Testing

The project uses Pytest for backend testing.

Run the test suite:

```bash
pytest
```

For detailed output:

```bash
pytest -v
```

---

## 🔐 Configuration & Security

- Environment variables are used for application configuration.
- `.env` is excluded from Git tracking.
- CORS configuration controls permitted frontend origins.
- Database configuration supports local and production environments.
- External API requests use configurable timeout and retry settings.

---

## 🚀 Future Scope

Potential improvements include:

- Advanced climate trend detection algorithms.
- Automated environmental data collection.
- Interactive geographical data analysis.
- Anomaly detection in environmental observations.
- Long-term climate pattern comparison.
- Additional NASA Earth observation data sources.
- Improved analytical visualization support.
- Scalable cloud deployment.

---

## 👥 Team – Firestorm Innovators

**NASA Space Apps Challenge 2026**

We are a team of innovators working to explore environmental data through technology, scientific analysis, and software engineering.

Our goal is to make complex Earth system data easier to understand and analyze.

---

## 📜 Acknowledgements

We would like to acknowledge:

- NASA Space Apps Challenge.
- NASA POWER Project.
- Open-source Python community.
- FastAPI and scientific Python ecosystem.

---

**Built with ❤️ by Firestorm Innovators**

*Exploring Earth. Detecting Trends. Understanding Change.*