 RetailPulse

RetailPulse is a retail analytics project built around the Online Retail II transaction dataset. It combines exploratory analysis and forecasting/modeling notebooks with a Streamlit dashboard for reviewing sales, customer, product, and inventory metrics.

## Features

- **Operations dashboard:** revenue and transaction KPIs, monthly sales trends, demand what-if simulation, RFM-based customer views and churn-risk tiers, inventory recommendations, and CSV/TXT exports.
- **Exploratory data analysis:** dataset structure, missing values and outliers, transaction and cancellation patterns, time trends, customer spending, product and basket metrics, geography, and correlations.
- **Analysis notebooks:** data ingestion and cleaning, EDA, customer segmentation with RFM and clustering, churn prediction, demand forecasting, and inventory optimization.
- **Included data artifacts:** raw transaction data and processed datasets used by the analyses.

## Repository layout

```text
RetailPulse/
├── dashboard/
│   └── dashboard.py
├── data/
│   ├── raw/
│   │   └── online_retail_II.csv
│   └── processed/
│       ├── cleaned_retail.csv
│       ├── churn_predictions.csv
│       ├── customer_segments.csv
│       ├── daily_sales.csv
│       ├── demand_forecast.csv
│       └── inventory_optimization_holt.csv
├── notebooks/
│   ├── Data Ingestion & Cleaning.ipynb
│   ├── EDA.ipynb
│   ├── Customer Segmentation (RFM + Clustering).ipynb
│   ├── Customer Churn Prediction.ipynb
│   ├── 03_Demand_forecasting.ipynb
│   └── Inventory Optimisation.ipynb
└── README.md
```

## Run the dashboard

Python 3.9 or newer is recommended. From the repository root, create and activate a virtual environment, then install the dashboard packages:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the dependencies and start Streamlit:

```bash
python -m pip install streamlit pandas numpy matplotlib seaborn
streamlit run dashboard/dashboard.py
```

The dashboard loads `data/processed/cleaned_retail.csv` when that file is available. If it cannot find the processed file, it attempts to create a synthetic sample dataset as a fallback. For normal use, run the command from the repository root so the included processed data is found.

## Explore the notebooks

Open the notebooks in JupyterLab or another Jupyter-compatible environment:

```bash
python -m pip install jupyter
jupyter lab
```

Notebook dependencies vary by analysis. In addition to pandas, NumPy, Matplotlib, and Seaborn, the notebooks use packages such as scikit-learn and statsmodels. The demand forecasting notebook also imports Prophet and TensorFlow. Install the packages required by the notebook you plan to run.

Run the data ingestion and cleaning notebook before downstream analyses if you want to reproduce the processed data from the raw CSV. Notebooks expect project data under `data/`; if a notebook uses a different relative path in your environment, update that path to match the repository layout.

## Data

The repository includes `data/raw/online_retail_II.csv` and derived files under `data/processed/`. The dashboard derives additional fields such as revenue, year-month, hour, weekday, and a cancellation flag from the cleaned transaction data. Review the relevant notebook before regenerating processed datasets, since those files are outputs of the analysis workflow.

## Notes

- Dashboard what-if forecasts and operational alert text are illustrative dashboard logic; treat them as exploratory outputs rather than production forecasts or live inventory alerts.
- Churn tiers in the dashboard are assigned from recency thresholds, while the separate churn prediction notebook contains model-based analysis.
- The dashboard's inventory recommendation is a simple rule based on historical quantity and is not a substitute for a replenishment system.

## License

No license file is currently included. Add a license before distributing or reusing this project if you intend to grant permissions to others.
Filter files
+92
