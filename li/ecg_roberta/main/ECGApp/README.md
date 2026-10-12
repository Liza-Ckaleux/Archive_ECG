# ECG Analysis Application

This Django application allows users to upload ECG files in EDF format and analyze them. The application displays ECG signals from 8 leads (I, II, V1-V6) and provides a diagnosis based on the uploaded data.

## Installation (Windows)

1. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
mkdir media
mkdir media\ecg_files
```

4. Run migrations:
```bash
python manage.py migrate
```

5. Create a superuser (optional):
```bash
python manage.py createsuperuser
```

## Running the Application

1. Make sure you're in the virtual environment (you should see `(venv)` at the beginning of your command prompt)

2. Start the development server:
```bash
python manage.py runserver
```

3. Open your browser and navigate to:
```
http://127.0.0.1:8000/
```

## Usage

1. Click the "Upload ECG File" button
2. Select an EDF file containing ECG data
3. Click "Upload and Analyze"
4. View the results including:
   - ECG plots for all 8 leads
   - Diagnosis result

## Requirements

- Python 3.8+
- Django 5.0.2
- MNE 1.6.1
- NumPy 1.26.3
- Pandas 2.1.4
- Matplotlib 3.8.2

## Troubleshooting

If you encounter any issues:

1. Make sure all dependencies are installed correctly
2. Check that the virtual environment is activated
3. Ensure the media directories exist
4. Verify that you have write permissions in the project directory 