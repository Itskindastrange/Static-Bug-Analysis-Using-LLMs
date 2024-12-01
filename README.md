# Static Bug Analysis using BERT & Gemini API

This project performs **static bug analysis** on code snippets by using a **BERT-based model** to detect vulnerabilities and danger levels, followed by querying the **Gemini API** for suggested fixes.

## Requirements

- **Python 3.6+**
- **PyTorch**
- **Transformers** (for BERT)
- **Scikit-learn**
- **NumPy**
- **tqdm**
- **requests** (for Gemini API interaction)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset (`data.json`) should have the following columns:
- `specific_code`: Code snippets for analysis.
- `vulnerability_type`: Comma-separated list of vulnerabilities (multi-label).
- `danger_level`: A single danger level (e.g., `Critical`, `High`).

## Setup

1. **Train the Model**: Train the model using the provided training script:
   
   ```bash
   python train.py
   ```

2. **Predict Vulnerabilities & Danger Level**:
   
   Example code to predict vulnerabilities and get suggestions:

   ```python
   test_code = """
   # Vulnerable code: SQL Injection
   @app.route('/get_user', methods=['GET'])
   def get_user():
       user_id = request.args.get('user_id')
       conn = sqlite3.connect('example.db')
       cursor = conn.cursor()
       query = f"SELECT * FROM users WHERE id = {user_id}"
       cursor.execute(query)
       user = cursor.fetchone()
       return str(user)
   """
   vul_labels, danger_label = predict(test_code, model, tokenizer)
   print("Predicted Vulnerabilities:", vul_labels)
   print("Predicted Danger Level:", danger_label)
   ```

3. **Query Gemini API for Fix Suggestions**:

   Add you API Key to get suggestions

## Training & Fine-Tuning

- **BERT-based Model**: Fine-tuned to detect vulnerabilities and classify danger levels.
- **Dynamic Thresholding**: Adjusts detection thresholds based on model performance.


