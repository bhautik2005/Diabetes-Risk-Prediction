# 1. Create virtual environment
python -m venv .venv

# 2. Active Environment
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset from Kaggle
kaggle datasets download mathchi/diabetes-data-set
unzip diabetes-data-set.zip -d data/raw/

# 5. Run notebooks in order(Optional)
jupyter notebook notebooks/

# 6. Run Pipline of ml
python main.py

# 7. Launch prediction UI
streamlit run app/predict_ui.py
