# Livestock Disease Detection System

A student project for preliminary screening of cattle photographs. It combines a MobileNetV2-based CNN image classifier with a Flask web application.

## Supported classes

- Foot-and-mouth disease (FMD)
- Healthy
- Infectious bovine keratoconjunctivitis (IBK)
- Lumpy skin disease (LSD)
- Mastitis
- Ringworm

## Features

- Upload a cattle image through the web interface.
- Display the predicted class and model confidence score.
- Show general precautionary guidance for the predicted class.
- Store prediction history in SQLite and view it in the history page.

## Technology stack

Python, TensorFlow/Keras, MobileNetV2, Flask, SQLite, HTML, CSS, and JavaScript.

## Project structure

```text
D:\LivestockCare\
├── app.py
├── requirements.txt
├── .python-version
├── database/
│   └── view_database.py
├── models/
│   ├── class_names.json
│   └── livestock_disease_model.keras
├── notebooks/
├── static/
│   ├── images/cattle-hero.png
│   ├── script.js
│   └── style.css
└── templates/
    ├── detection.html
    ├── history.html
    ├── index.html
    └── result.html
```

## Run on Windows

Open PowerShell in the existing project folder and run:

```powershell
Set-Location 'D:\LivestockCare'
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

If PowerShell blocks virtual-environment activation, use this for the current terminal session, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Open `http://127.0.0.1:5000` in a browser. The application creates its `uploads` folder and SQLite database in `database/livestock_app.db` when it starts. Set `FLASK_SECRET_KEY` in the environment if you need a stable session key across restarts; otherwise the application generates a temporary key at startup. Debug mode is off by default; set `$env:FLASK_DEBUG = "1"` only when debugging locally.

## Dataset and evaluation

The image dataset is not included. The training notebooks are included for reference. The trained model and class-name file are included because the web application requires them for predictions.

An earlier evaluation reported **80.65% test accuracy**. A later dataset check found exact duplicate images across the training, validation, and test subsets. Treat this accuracy as preliminary; it should be recalculated after removing duplicates and repartitioning the dataset.

## Important notice

This application provides preliminary image-based screening only. A prediction and its confidence score do not confirm a disease or prescribe treatment. Contact a veterinarian for diagnosis and care advice.