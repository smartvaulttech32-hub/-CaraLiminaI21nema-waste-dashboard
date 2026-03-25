CMT 444: Federated Learning for Kenyan Waste Management

 Project Overview
This project implements a **Distributed Machine Learning system** for waste management across 5 Kenyan counties using **Federated Learning (FedAvg)**. The system integrates official **NEMA 2024** waste management regulations and provides real-time AI predictions for waste optimization.

 Key Features
- Machine Learning Models** - Random Forest & Gradient Boosting for predictions
- Federated Learning** - FedAvg aggregation across 5 counties
- Privacy Preserved** - Raw data never leaves county systems
- NEMA 2024 Compliance** - Based on official Kenyan regulations
- Clean Cities Ranking** - 0-100% cleanliness scoring
- Research Section** - Best practices & global success stories

 Counties Covered
| County | Population | Smart Bins | Clean Score | Status |
|--------|------------|------------|-------------|--------|
| Kiambu | 2,417,000 | 40 | 75% | Clean ✅ |
| Kisumu | 1,155,000 | 20 | 72% | Clean ✅ |
| Nakuru | 2,162,000 | 35 | 68% | Moderate ⚠️ |
| Nairobi | 4,397,000 | 50 | 65% | Moderate ⚠️ |
| Mombasa | 1,208,000 | 25 | 58% | Dirty 🔴 |

AI/ML Models
| Model | Purpose | Algorithm | Accuracy |
|-------|---------|-----------|----------|
| Fill Level | Predict bin fill % | Random Forest | 85-92% R² |
| Clean Score | Predict city cleanliness | Random Forest | 85-92% R² |
| Overflow | Predict hours to overflow | Gradient Boosting | MAE: 0.8 hours |


### Federated Learning Process

| Step | Description |
|------|-------------|
| **1. Local Training** | Each county trains a model on its LOCAL data |
| **2. Weight Sharing** | Only model weights (not data) are shared |
| **3. FedAvg Aggregation** | Server averages weights: `Global = Σ(nᵢ/N) × Modelᵢ` |
| **4. Global Model** | New model benefits all counties without seeing raw data |

### Privacy Guarantee
- ❌ Raw waste data NEVER leaves county systems
- ✅ Only model weights (feature importance) are shared
- ✅ Federated Averaging (FedAvg) preserves privacy
- ✅ Each county maintains full data sovereignty

### Federated Learning Results
| Metric | Value |
|--------|-------|
| Active Clients | 5/5 |
| Global Model Weight | 0.13 |
| Privacy Preserved | 100% |

##  Live Demo
- **Streamlit Dashboard:** [Click Here](https://ctajdiafb7f5lxdx7ohexf.streamlit.app/)

## NEMA 2024 Data Integration
| Parameter | Value | Source |
|-----------|-------|--------|
| National Daily Waste | 22,000 tons | NEMA PDF Page 1 |
| Per Capita Waste | 0.5 kg/day | NEMA PDF Page 1 |
| Critical Threshold | 75% | NEMA Section 7.2 |
| Collection Target | 85% | NEMA PDF Page 4 |

##  Tech Stack
- **Frontend:** Streamlit
- **Visualization:** Plotly
- **ML Models:** scikit-learn (Random Forest, Gradient Boosting)
- **Data Processing:** Pandas, NumPy
- **Deployment:** Streamlit Cloud

## 📁 Project Structure
Nema-waste-management/
├── waste_dashboard.py # Main application code
├── requirements.txt # Python dependencies
├── README.md # This file
└── .gitignore # Git ignore file
