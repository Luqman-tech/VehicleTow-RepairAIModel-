# Predictive AI Model for Equitable Distribution of Auto Repair Resources

**Group 30 - AI for Software Development**  
**SDG Focus:** SDG 9 - Industry, Innovation, and Infrastructure

## Overview

This project addresses the critical infrastructure gap in motor vehicle repair and towing services, particularly in underdeveloped and rural areas. Using machine learning, we've developed a predictive model that forecasts the number of ASE-certified mechanics needed based on location and service request characteristics.

## Problem Statement

Many communities lack sufficient access to ASE-certified mechanics and properly distributed repair facilities, creating:
- Infrastructure inefficiencies
- Prolonged vehicle downtimes
- Unequal access to critical industrial services
- Negative impacts on mobility and logistics networks

## Solution Approach

Our AI/ML solution employs data analysis and predictive modeling to:
- Identify geographic disparities in auto repair service distribution
- Predict mechanic resource needs based on location attributes
- Support data-driven workforce planning and service placement decisions

## Dataset

The project uses a **Motor Vehicle Repair and Towing** services dataset containing:
- ZIP codes and geographic information
- City and state data
- Number of ASE-certified mechanics
- Service request timestamps
- Vehicle information and problem descriptions

## Model Architecture

### Features Used:
- **Temporal Features:** Hour of request, day of week
- **Vehicle Characteristics:** Vehicle age
- **Location Data:** Encoded location information
- **Service Type:** Encoded problem descriptions

### Algorithm:
- **Linear Regression** for predicting the number of certified mechanics
- **Label Encoding** for categorical variables (location, problem type)

### Tools & Libraries:
- Python
- Pandas (data manipulation)
- Scikit-learn (machine learning)
- NumPy (numerical computing)
- Seaborn & Matplotlib (visualization)
- Flask (web application)
- Joblib (model persistence)

## Project Structure

```
project/
├── aimodel.py              # Model training script
├── app.py                  # Flask web application
├── model.joblib            # Trained model (generated)
├── le_location.joblib      # Location label encoder (generated)
├── le_problem.joblib       # Problem label encoder (generated)
├── Motor_Vehicle_Repair_and_Towing.csv  # Dataset
├── templates/
│   └── index.html          # Web interface template
└── README.md               # This file
```

## Installation & Setup

### Prerequisites
- Python 3.7+
- Required packages (install via pip):

```bash
pip install pandas numpy scikit-learn flask joblib
```

### Running the Project

1. **Train the Model:**
   ```bash
   python aimodel.py
   ```
   This will generate the model and encoder files.

2. **Launch the Web Application:**
   ```bash
   python app.py
   ```
   The application will be available at `http://localhost:5000`

## Usage

### Web Interface
The Flask application provides a user-friendly interface where users can input:
- **Request Time:** When the service is needed
- **Vehicle Age:** Age of the vehicle requiring service
- **Location:** Service location
- **Problem Description:** Type of repair needed

The model will predict the optimal number of ASE-certified mechanics needed for that scenario.

### Model Training
The `aimodel.py` script handles:
- Data loading and cleaning
- Feature engineering (time-based features)
- Label encoding for categorical variables
- Model training and validation
- Saving trained model and encoders

## Key Findings

Our analysis revealed:
- **Geographic Disparities:** Significant variation in mechanic availability across ZIP codes
- **Urban Saturation:** Over-concentration of services in urban areas
- **Rural Underservice:** Many rural locations lack adequate coverage
- **State-Level Inconsistencies:** Uneven distribution patterns across different states

## Ethical Considerations

### Data Bias
- Dataset primarily concentrated in Maryland, potentially limiting generalizability
- Recommendations include expanding data collection to other regions

### Equity & Access
- Model designed to identify and address existing disparities
- Focus on promoting equitable service distribution

### Privacy
- No personally identifiable information (PII) used
- Maintains ethical standards in data handling

## Impact & Applications

This model supports **SDG 9 (Industry, Innovation, and Infrastructure)** by enabling:

- **Policy Decision Support:** Data-driven insights for government planning
- **Investment Guidance:** Strategic placement of training programs and service centers
- **Workforce Planning:** Optimal allocation of certified mechanics
- **Community Development:** Improved access to essential automotive services

## Future Enhancements

- Expand dataset to include more geographic regions
- Incorporate additional features (economic indicators, population density)
- Implement advanced algorithms (ensemble methods, neural networks)
- Develop mobile application for field use
- Add real-time data integration capabilities

## Contributing

This project demonstrates how AI can address practical infrastructure challenges. Contributions are welcome to:
- Improve model accuracy
- Expand geographic coverage
- Enhance user interface
- Add new features and functionality

## License

This project is developed for educational and research purposes as part of the AI for Software Development program.

## Contact

**Group 30**  
AI for Software Development Program

---

*This project proves that AI, when thoughtfully applied, is a true engine of sustainable progress in addressing infrastructure challenges.*
# VehicleTow-RepairAIModel-
