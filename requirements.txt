# SMS Spam Classifier API

A FastAPI-powered SMS spam classifier built for **Task 05**. It uses an improved Bidirectional LSTM (BiLSTM) model to classify a text message as **ham** or **spam**, returning both a spam probability and a confidence value.

## Project highlights

- Trained on the UCI SMS Spam Collection dataset
- Text preprocessing: lowercasing and punctuation removal, while preserving digits
- Tokenizer-based text encoding with padded sequences of length 100
- 80/20 stratified train/test split
- FastAPI endpoint: `POST /predict`
- Interactive API testing available through FastAPI's `/docs` page

## Dataset

The original dataset contained **5,572** messages. After removing **403** duplicates, the final dataset contained **5,169** messages:

| Class | Messages |
| --- | ---: |
| Ham | 4,516 |
| Spam | 653 |
| **Total** | **5,169** |

## Model performance

Two neural-network approaches were evaluated:

| Model | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: |
| Initial neural network | 97.49% | 95.65% | 83.97% | 89.43% |
| Improved BiLSTM | **98.55%** | **96.03%** | **92.37%** | **94.16%** |

The improved BiLSTM was selected as the final model.

### Confusion matrix

```text
[[898,   5],
 [ 10, 121]]
```

## Project structure

```text
sms-spam-classifier/
├── app.py              # FastAPI application
├── best_model.keras    # Trained BiLSTM model
├── tokenizer.pkl       # Saved tokenizer
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── .gitignore          # Files Git should not upload
```

## Requirements

- Python 3.x
- TensorFlow 2.21.0
- FastAPI 0.115.12
- Uvicorn 0.34.2
- NumPy 2.1.3

Put the following in `requirements.txt`:

```text
tensorflow==2.21.0
fastapi==0.115.12
uvicorn==0.34.2
numpy==2.1.3
```

## Run locally

1. Create and activate a virtual environment.

   **Windows PowerShell**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies.

   ```powershell
   pip install -r requirements.txt
   ```

3. Start the API.

   ```powershell
   uvicorn app:app --reload
   ```

4. Open `http://127.0.0.1:8000/docs` in a browser and test the endpoint.

The API was verified locally and tested through the interactive `/docs` page with a correct result.

## API usage

### `POST /predict`

Send an SMS message to receive its predicted label, spam probability, and confidence.

Example request body:

```json
{
  "message": "Congratulations! You have won a prize. Reply now to claim it."
}
```

Example response shape:

```json
{
  "prediction": "spam",
  "spam_probability": 0.0,
  "confidence": 0.0
}
```

`spam_probability` and `confidence` are returned by the API as numeric values.

## Files to keep private or exclude

Do **not** upload your virtual environment or secret values. Add this to `.gitignore`:

```gitignore
.venv/
__pycache__/
*.py[cod]
.env
```

Before uploading `best_model.keras`, check its size. GitHub blocks individual files larger than **100 MB**. If it exceeds that limit, use Git LFS or keep the model artifact outside the repository and document how to obtain it.

## Tech stack

- Python
- TensorFlow / Keras
- Bidirectional LSTM
- FastAPI
- Uvicorn

## Future improvements

- Add a simple user interface for message classification
- Add model-version information to API responses
- Expand the dataset and retrain periodically

