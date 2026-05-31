# Static Bug Analysis Agent

A two-stage static code analysis pipeline that detects vulnerabilities 
and suggests fixes — without running the code.

**Stage 1:** A fine-tuned BERT model classifies code snippets by 
vulnerability type (multi-label) and danger level (Critical, High, etc.)

**Stage 2:** Detected vulnerabilities are passed to the Gemini API, 
which generates targeted fix suggestions for each issue found.

---

## How It Works

Code Snippet → BERT Classifier → Vulnerability Labels + Danger Level
↓
Gemini API → Fix Suggestions


---

## Example

Input — a SQL injection vulnerability:

```python
query = f"SELECT * FROM users WHERE id = {user_id}"
```

Output:
Predicted Vulnerabilities: ['SQL Injection', 'Input Validation']
Predicted Danger Level: Critical
Suggested Fix: Use parameterized queries — cursor.execute(
"SELECT * FROM users WHERE id = ?", (user_id,))


---

## Stack

- **Model:** BERT (fine-tuned, PyTorch + HuggingFace Transformers)
- **Classification:** Multi-label vulnerability detection + danger level classification
- **Fix Generation:** Gemini API
- **Training:** Dynamic thresholding based on model performance

---

## Setup

```bash
pip install -r requirements.txt
```

1. Train the model:
```bash
python train.py
```

2. Run predictions on your code snippet — see `predict.py`

3. Add your Gemini API key to `.env` to enable fix suggestions

---

## Dataset Format

`data.json` expects:

| Field | Description |
|---|---|
| `specific_code` | Raw code snippet |
| `vulnerability_type` | Comma-separated vulnerability labels |
| `danger_level` | Single label: Critical, High, Medium, Low |

---

## Built By

[Abdullah Ahmad](https://abdullahahmaddd.vercel.app) · 
[LinkedIn](https://linkedin.com/in/abdullahahmd)
