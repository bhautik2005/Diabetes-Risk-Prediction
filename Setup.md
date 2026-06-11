# 1. Create virtual environment
--> python -m venv .venv
Active Environment--> .\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle
kaggle datasets download mathchi/diabetes-data-set
unzip diabetes-data-set.zip -d data/raw/

# 4. Run notebooks in order(Optional)
jupyter notebook notebooks/

# 5. Run Pipline of ml
python main.py

# 5. Launch prediction UI
streamlit run app/predict_ui.py
