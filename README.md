# 📊 EM-4 Statistical Methods Project

## 🚀 Overview
This project is developed as part of **Engineering Mathematics-IV (EM-4)** and focuses on implementing important **statistical and optimization techniques** using Python.

The system provides an **interactive dashboard** for analyzing data using:
- Spearman Rank Correlation
- Parabolic Curve Fitting

It also includes graphical visualization and real-time computation using Streamlit.

---

## 📌 Assignment Details
- **Subject:** Engineering Mathematics-IV (BSC07)  
- **Semester:** S.E. Semester IV – INFT  
- **Faculty:** Dr. Uday Kashid  
- **Assignment No:** 9  
- **Title:** Mini Project – Statistical & Optimization Code Challenge  

---

## 🎯 Objective
The objective of this project is to implement statistical methods using Python:

- To compute **Spearman Rank Correlation** for measuring relationships between variables  
- To perform **Parabolic Curve Fitting** using Least Squares Method  
- To visualize data using graphs and interactive UI  
- To analyze real-world datasets efficiently  

(As described in the project document :contentReference[oaicite:0]{index=0})

---

## 📖 Project Description

### 🔹 1. Spearman Rank Correlation
- Converts raw data into ranks  
- Handles tied values using average ranking  
- Computes rank differences and correlation coefficient  
- Uses both standard and corrected formulas  

✔ Helps determine:
- Strength of relationship  
- Direction (positive/negative)

---

### 🔹 2. Parabolic Curve Fitting
- Fits data into quadratic equation:  
  **y = a + bx + cx²**
- Uses **Least Squares Method**
- Solves normal equations using matrix method  
- Calculates:
  - Coefficients (a, b, c)
  - Residuals
  - R² (goodness of fit)

---
## ⚙️ Methodology

### 🔹 Spearman Rank Correlation

1. Input data is collected (CSV or manual input)  
2. Data is converted into ranks  
3. Ties are handled using average ranking  
4. Rank differences (d) are calculated  
5. Square of differences (d²) is computed  
6. Correlation coefficient is calculated using:

R = 1 − (6 Σd²) / (n(n² − 1))

7. If ties exist, correction factor is applied  

---

### 🔹 Parabolic Curve Fitting

1. Input dataset (x, y) is taken  
2. Required columns are computed: x², x³, x⁴, xy, x²y  
3. Normal equations are formed  
4. Matrix method is used to solve coefficients (a, b, c)  
5. Curve equation formed: y = a + bx + cx²  
6. Residuals and R² value are calculated
---
## 💻 Code Explanation

### 🔹 Spearman Module (spearman.py)

- `assign_ranks()` → Assigns ranks with tie handling  
- `compute_spearman()` → Calculates correlation coefficient  
- `plot_spearman()` → Generates 6-panel visualization  

---

### 🔹 Parabola Module (parabola.py)

- `build_table()` → Creates required mathematical columns  
- `compute_parabola()` → Solves normal equations  
- `plot_parabola()` → Displays curve fitting graphs  

---

### 🔹 Main App (app.py)

- Built using Streamlit  
- Handles user input (CSV/manual)  
- Calls respective modules  
- Displays results and graphs
---

## 🚀 Features

- 📂 Upload CSV file or enter data manually  
- 📊 Automatic rank calculation (with tie handling)  
- 📈 6-panel graphical visualization  
- 🧮 Step-by-step calculations  
- 🎯 Clean UI using Streamlit  
- ⚡ Real-time computation  

---

## 💡 Key Concepts Covered

- Spearman Rank Correlation (with tie correction)  
- Least Squares Method  
- Regression Analysis  
- Data Visualization  
- Interactive Web Applications  

---

## 🛠️ Technologies Used

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Streamlit  

---

## ▶️ Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/dhanrajmogera-hue/Maths-.git
cd Maths-
```
### 2️⃣ Install Dependencies
```bash
pip install streamlit numpy pandas matplotlib
```
### 3️⃣ Run the Application
```bash
streamlit run app.py
```
---

## 🤝 Contributions

### 👨‍💻 Swapnil Bhabal
- Improved overall documentation and added detailed comments  
- Enhanced code readability and structure  
- Organized project workflow and repository  

### 👨‍💻 Dhanraj Mogera
- Initialized the project repository  
- Provided core project structure  
- Managed initial file uploads  

### 👨‍💻 Vishal Prajapati
- Managed file structure and repository updates  
- Performed code modifications and cleanup  
- Maintained project consistency  

### 👨‍💻 Altamash Ansari
- Implemented Parabolic Curve Fitting module  
- Added graphical visualizations  
- Improved analytical representation  

### 👩‍💻 Sharwari Lohate
- Added and updated CSV datasets  
- Improved dataset formatting and structure  
- Assisted in data validation

### 👨‍💻 Meet Alshi
- Performed testing and validation  
- Assisted in dataset verification  
- Supported debugging and improvements
   
