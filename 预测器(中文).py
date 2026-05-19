import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import os
import sys
import sklearn


# ===================== 配置部分 =====================
# 模型文件路径（支持EXE打包后的相对路径）
def resource_path(relative_path):
    """获取资源文件的绝对路径，兼容打包后的EXE"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


# 模型目录（打包时建议把模型文件和EXE放在同一目录下的models文件夹里）
#MODEL_DIR = r"C:\Users\29546\Desktop\建强的模型\XGBoost\运行结果2\模型参数"
# 打包时请把三个模型文件和EXE放在同一目录下，用下面的路径
MODEL_DIR = resource_path("models")

MODEL_PATHS = {
    "黏聚力(kPa)": os.path.join(MODEL_DIR, "黏聚力_完整模型.pkl"),
    "内摩擦角(°)": os.path.join(MODEL_DIR, "内摩擦角_完整模型.pkl"),
    "剪切模量(MPa)": os.path.join(MODEL_DIR, "剪切模量_完整模型.pkl")
}

# 输入参数名称（和训练数据顺序一致）
INPUT_PARAM_NAMES = [
    "bfemod(MPa)", "ssemod(MPa)", "ssfric", "rremod(MPa)", "rrfric", "rrpb_ten(kPa)", "VBP(0.2-0.8)"
]
OUTPUT_PARAM_NAMES = ["黏聚力(kPa)", "内摩擦角(°)", "剪切模量(MPa)"]

# ===================== 加载模型 =====================
models = {}
try:
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到模型文件：{path}\n请确保模型文件和程序在同一目录下的models文件夹中！")
        models[name] = joblib.load(path)
except Exception as e:
    messagebox.showerror("模型加载失败", f"错误信息：{str(e)}")
    sys.exit()  # 替换 exit(1) 为 sys.exit(1)


# ===================== 预测函数 =====================
def predict_all():
    try:
        # 读取并校验输入参数
        input_values = []
        for entry in input_entries:
            val_str = entry.get().strip()
            if not val_str:
                raise ValueError("请输入所有参数的值！")
            val = float(val_str)
            input_values.append(val)

        # 转换为模型输入格式 (1, 7)
        X = np.array(input_values).reshape(1, -1)

        # 调用三个模型预测
        results = {}
        for name, model in models.items():
            pred = model.predict(X)[0]
            results[name] = pred

        # 更新结果显示
        for i, name in enumerate(OUTPUT_PARAM_NAMES):
            result_labels[i].config(text=f"{name}预测值：{results[name]:.4f}")

    except ValueError as e:
        messagebox.showwarning("输入错误", f"请输入有效的数字！\n错误：{str(e)}")
    except Exception as e:
        messagebox.showerror("预测失败", f"错误信息：{str(e)}")


# ===================== 清空输入函数 =====================
def clear_inputs():
    for entry in input_entries:
        entry.delete(0, tk.END)
    for label in result_labels:
        label.config(text=f"{label.cget('text').split('：')[0]}：")


# ===================== 构建UI界面 =====================
root = tk.Tk()
root.title("土石混合体PFC2D离散元力学参数预测平台")
root.geometry("950x750")
root.resizable(True, True)

# 设置全局样式
style = ttk.Style(root)
style.theme_use("clam")
style.configure("Accent.TButton", font=("微软雅黑", 15, "bold"), padding=10)
style.configure("Title.TLabel", font=("微软雅黑", 20, "bold"), foreground="#2c3e50")
style.configure("Result.TLabel", font=("微软雅黑", 15, "bold"), foreground="#2980b9")
style.configure("Section.TLabelframe.Label", font=("微软雅黑", 15, "bold"), foreground="#34495e")

# 标题
title_label = ttk.Label(root, text="细观参数 → 力学参数预测", style="Title.TLabel")
title_label.pack(pady=20)

# 输入参数框架
input_frame = ttk.LabelFrame(root, text="7个细观参数输入", style="Section.TLabelframe", padding=(20, 15))
input_frame.pack(padx=30, pady=10, fill="x")

input_entries = []
for i, param_name in enumerate(INPUT_PARAM_NAMES):
    row = i // 2
    col = i % 2
    label = ttk.Label(input_frame, text=f"{param_name}：", font=("微软雅黑", 15))
    label.grid(row=row, column=col * 2, padx=15, pady=5, sticky="w")
    entry = ttk.Entry(input_frame, font=("微软雅黑", 15), width=18)
    entry.grid(row=row, column=col * 2 + 1, padx=5, pady=5)
    input_entries.append(entry)

# 按钮框架
button_frame = ttk.Frame(root)
button_frame.pack(pady=20)

predict_btn = ttk.Button(button_frame, text="开始预测", command=predict_all, style="Accent.TButton")
predict_btn.grid(row=0, column=0, padx=15)

clear_btn = ttk.Button(button_frame, text="清空输入", command=clear_inputs, style="Accent.TButton")
clear_btn.grid(row=0, column=1, padx=15)

# 结果显示框架
result_frame = ttk.LabelFrame(root, text="预测结果", style="Section.TLabelframe", padding=(20, 15))
result_frame.pack(padx=20, pady=5, fill="x")

result_labels = []
for name in OUTPUT_PARAM_NAMES:
    label = ttk.Label(result_frame, text=f"{name}预测值：", style="Result.TLabel")
    label.pack(pady=12, anchor="w")
    result_labels.append(label)

# 底部说明
tip_label = ttk.Label(root, text="提示：输入参数需满足量纲、取值范围要求：bfemod和ssemod:100-10000，ssemod:10-1000，rrpb_ten:100-5000，VBP:0.2-0.8，ssfric和rrfric:0-1",
                      font=("微软雅黑", 10), foreground="#F67C6F")
tip_label.pack(pady=10)

# 运行主循环
if __name__ == "__main__":
    root.mainloop()
