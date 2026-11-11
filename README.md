# Logistic Regression - Diabetes Prediction

This project is prepared to open and run directly in Visual Studio Code.

## Project contents
- `logistic_regression.py` - complete runnable Python program
- `data/diabetes.csv` - dataset used by the program
- `Logistic_Regression.ipynb` - notebook version for cell-by-cell learning
- `requirements.txt` - required Python packages
- `SETUP_AND_VIVA_GUIDE.pdf` - VS Code setup, run steps, and viva revision
- `outputs/` - generated evaluation charts and coefficients (created after running)

## One-time setup in VS Code
1. Install Python 3.10 or newer.
2. Install Visual Studio Code and the Microsoft Python extension.
3. Extract this ZIP file.
4. In VS Code, select **File > Open Folder** and open the `Logistic-Regression` folder.
5. Open **Terminal > New Terminal**.
6. Create a virtual environment:
   - Windows: `py -m venv .venv`
   - If `py` is unavailable: `python -m venv .venv`
7. Activate it:
   - PowerShell: `.venv\Scripts\Activate.ps1`
   - Command Prompt: `.venv\Scripts\activate.bat`
8. Install dependencies: `python -m pip install -r requirements.txt`
9. Select the `.venv` interpreter: press `Ctrl+Shift+P`, choose **Python: Select Interpreter**, select the interpreter inside `.venv`.
10. Run: `python logistic_regression.py`

If PowerShell blocks activation, use Command Prompt terminal and run `.venv\Scripts\activate.bat`, or run the script with `.venv\Scripts\python.exe logistic_regression.py`.

## Expected output
The terminal prints the dataset preview, train/test sizes, accuracy, ROC-AUC, classification report, and confusion matrix. The program creates:
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`
- `outputs/model_coefficients.csv`

Exact metric values are printed by your local run. Do not memorize a result from another computer.

## Important
This is an educational model using the supplied diabetes dataset. It is not a medical diagnostic tool.
