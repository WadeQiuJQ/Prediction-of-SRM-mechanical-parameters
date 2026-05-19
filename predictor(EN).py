import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import os
import sys
import sklearn


# ===================== Configuration Section =====================
# Model file path (supports relative path after EXE packaging)
def resource_path(relative_path):
    """Get the absolute path of resource files, compatible with packaged EXE"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


# Model directory (it is recommended to put model files and EXE in the models folder under the same directory when packaging)
#MODEL_DIR = r"C:\Users\29546\Desktop\建强的模型\XGBoost\运行结果2\模型参数"
# When packaging, please put the three model files and EXE in the same directory, use the following path
MODEL_DIR = resource_path("models")

MODEL_PATHS = {
    "c(kPa)": os.path.join(MODEL_DIR, "黏聚力_完整模型.pkl"),
    "φ(°)": os.path.join(MODEL_DIR, "内摩擦角_完整模型.pkl"),
    "G(MPa)": os.path.join(MODEL_DIR, "剪切模量_完整模型.pkl")
}

# Input parameter names (consistent with the order of training data)
INPUT_PARAM_NAMES = [
    "bfemod(MPa)", "ssemod(MPa)", "ssfric", "rremod(MPa)", "rrfric", "rrpb_ten(kPa)", "VBP(0.2-0.8)"
]
OUTPUT_PARAM_NAMES = ["c(kPa)", "φ(°)", "G(MPa)"]

# ===================== Load Models =====================
models = {}
try:
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}\nPlease ensure the model files are in the 'models' folder in the same directory as the program!")
        models[name] = joblib.load(path)
except Exception as e:
    messagebox.showerror("Model Loading Failed", f"Error message: {str(e)}")
    sys.exit()  # Replace exit(1) with sys.exit(1)


# ===================== Prediction Function =====================
def predict_all():
    try:
        # Read and verify input parameters
        input_values = []
        for entry in input_entries:
            val_str = entry.get().strip()
            if not val_str:
                raise ValueError("Please enter values for all parameters!")
            val = float(val_str)
            input_values.append(val)

        # Convert to model input format (1, 7)
        X = np.array(input_values).reshape(1, -1)

        # Call three models for prediction
        results = {}
        for name, model in models.items():
            pred = model.predict(X)[0]
            results[name] = pred

        # Update result display
        for i, name in enumerate(OUTPUT_PARAM_NAMES):
            result_labels[i].config(text=f"{name} Prediction Value: {results[name]:.4f}")

    except ValueError as e:
        messagebox.showwarning("Input Error", f"Please enter valid numbers!\nError: {str(e)}")
    except Exception as e:
        messagebox.showerror("Prediction Failed", f"Error message: {str(e)}")


# ===================== Clear Input Function =====================
def clear_inputs():
    for entry in input_entries:
        entry.delete(0, tk.END)
    for label in result_labels:
        label.config(text=f"{label.cget('text').split(':', 1)[0]}:")


# ===================== Build UI Interface =====================
root = tk.Tk()
root.title("Soil-rock Mixture PFC2D Discrete Element Mechanical Parameter Prediction Platform")
root.geometry("950x750")
root.resizable(True, True)

# Set global style
style = ttk.Style(root)
style.theme_use("clam")
style.configure("Accent.TButton", font=("Microsoft YaHei", 15, "bold"), padding=10)
style.configure("Title.TLabel", font=("Microsoft YaHei", 20, "bold"), foreground="#2c3e50")
style.configure("Result.TLabel", font=("Microsoft YaHei", 15, "bold"), foreground="#2980b9")
style.configure("Section.TLabelframe.Label", font=("Microsoft YaHei", 15, "bold"), foreground="#34495e")

# Title
title_label = ttk.Label(root, text="Mesoscopic Parameters → Mechanical Parameters Prediction", style="Title.TLabel")
title_label.pack(pady=20)

# Input parameter frame
input_frame = ttk.LabelFrame(root, text="7 Mesoscopic Parameters Input", style="Section.TLabelframe", padding=(20, 15))
input_frame.pack(padx=30, pady=10, fill="x")

input_entries = []
for i, param_name in enumerate(INPUT_PARAM_NAMES):
    row = i // 2
    col = i % 2
    label = ttk.Label(input_frame, text=f"{param_name}:", font=("Microsoft YaHei", 15))
    label.grid(row=row, column=col * 2, padx=15, pady=5, sticky="w")
    entry = ttk.Entry(input_frame, font=("Microsoft YaHei", 15), width=18)
    entry.grid(row=row, column=col * 2 + 1, padx=5, pady=5)
    input_entries.append(entry)

# Button frame
button_frame = ttk.Frame(root)
button_frame.pack(pady=20)

predict_btn = ttk.Button(button_frame, text="Start Prediction", command=predict_all, style="Accent.TButton")
predict_btn.grid(row=0, column=0, padx=15)

clear_btn = ttk.Button(button_frame, text="Clear Inputs", command=clear_inputs, style="Accent.TButton")
clear_btn.grid(row=0, column=1, padx=15)

# Result display frame
result_frame = ttk.LabelFrame(root, text="Prediction Results", style="Section.TLabelframe", padding=(20, 15))
result_frame.pack(padx=20, pady=5, fill="x")

result_labels = []
for name in OUTPUT_PARAM_NAMES:
    label = ttk.Label(result_frame, text=f"{name} Prediction Value:", style="Result.TLabel")
    label.pack(pady=12, anchor="w")
    result_labels.append(label)

# Bottom tip
tip_label = ttk.Label(
    root,
    text="Tip: Input parameters must meet value range: bfemod/ssemod:100-10000, rremod:10-1000, rrpb_ten:100-5000\nVBP:0.2-0.8, ssfric/rrfric:0-1",
    font=("Microsoft YaHei", 10),
    foreground="#F67C6F",
    wraplength=880,
    justify="left"
)
tip_label.pack(pady=10)

# Run main loop
if __name__ == "__main__":
    root.mainloop()