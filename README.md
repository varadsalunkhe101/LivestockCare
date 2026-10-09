# Livestock Disease Detection System

A student project that uses image classification to screen cattle photographs for selected disease categories. The system combines a MobileNetV2-based CNN model with a Flask web application.

## Supported Classes

- Foot-and-mouth disease (FMD)
- Infectious bovine keratoconjunctivitis (IBK)
- Lumpy skin disease (LSD)
- Mastitis
- Ringworm
- Healthy

## Features

- Upload a cattle image through a web interface.
- Predict one of the classes supported by the trained model.
- Display the predicted class and model confidence score.
- Show primary precautionary guidance associated with the prediction.
- Store prediction history in an SQLite database.
- View previous predictions through the history page.

## Technology Stack

- Python
- TensorFlow and Keras
- MobileNetV2 with transfer learning
- Flask
- SQLite
- HTML, CSS, and JavaScript

## Project Structure

```text
livestock_skin_disease_project/
├── app.py
├── requirements.txt
├── database/
│   └── view_database.py
├── models/
│   ├── class_names.json
│   └── livestock_disease_model.keras
├── notebooks/
├── static/
│   ├── images/
│   ├── script.js
│   └── style.css
└── templates/
    ├── detection.html
    ├── history.html
    ├── index.html
    └── result.html
```

## Setup and Run

1. Clone the repository and open its folder:

   ```powershell
   git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
   cd livestock_skin_disease_project
   ```

2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the required packages:

   ```powershell
   pip install -r requirements.txt
   ```

4. Confirm that these model files are present:

   ```text
   models/livestock_disease_model.keras
   models/class_names.json
   ```

5. Start the application:

   ```powershell
   python app.py
   ```

6. Open the local address printed in the terminal in a web browser.

## Dataset and Model

The image dataset is not included in this repository. The training notebooks are included as project references. The trained model and its class-name file are included because the web application needs them to make predictions.

## Evaluation Note

An earlier evaluation reported **80.65% test accuracy**. A later dataset check found exact duplicate images across the training, validation, and test subsets. Therefore, this result is preliminary and should be recalculated after removing duplicates and repartitioning the data.

## Important Notice

This application provides preliminary image-based screening only. Its prediction and confidence score do not confirm a disease. Contact a veterinarian for diagnosis and treatment advice.
