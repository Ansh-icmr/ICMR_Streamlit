# ICMR_Streamlit  
*A lightweight Streamlit app for RAG purposes*  
> “Just for the RAG purpose.”

---

## 📘 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the App](#running-the-app)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)

## 🧭 Overview
**ICMR_Streamlit** is a simple web application built with **Streamlit** — a Python framework for creating interactive web apps.  
This project appears to serve as a test or prototype, focusing on **PDF management and merging** tasks.

The app may process or display PDF files (such as those from ICMR), and offers an interactive front-end to perform such operations quickly.

---

## ✨ Features
- 🎨 Streamlit-based front-end (`streamlit_frontend_tool.py`)  
- 📄 PDF merging functionality (`merge.py`)  
- 📁 Organized storage for local PDFs (`pdfs/`) and URL-based PDFs (`icmr_url_pdf/`)  
- ⚙️ Lightweight and easy to deploy  
- 🧰 Simple setup — ideal for testing Streamlit workflows

---

## 🚀 Getting Started

### 🔧 Prerequisites
- Python **3.7+**
- `pip` (Python package manager)
- (Optional) A Streamlit Cloud or local environment to host the app

---

### ⚙️ Installation

Clone this repository:
bash
git clone https://github.com/Ansh-icmr/ICMR_Streamlit.git
cd ICMR_Streamlit
(Optional) Create a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Or, if dependencies are listed in packages.txt:

bash
Copy code
pip install -r packages.txt
▶️ Running the App
Run the Streamlit application:

bash
Copy code
streamlit run streamlit_frontend_tool.py
This will start a local server and open the app in your web browser.

If you want to run the merge utility manually:

bash
Copy code
python merge.py
📂 Project Structure
graphql
Copy code
ICMR_Streamlit/
├── .devcontainer/              # (Optional) Dev container configuration
├── indices/                    # Stores index or metadata files
├── icmr_url_pdf/               # PDFs from ICMR URLs
├── pdfs/                       # Local PDF files
├── merge.py                    # PDF merging logic
├── streamlit_frontend_tool.py  # Main Streamlit app
├── requirements.txt            # Dependencies
├── packages.txt                # Alternate dependency list
├── islr.pdf                    # Example PDF file
└── .gitignore
🧠 Usage
Place your PDF files inside the pdfs/ or icmr_url_pdf/ folders.

Run merge.py to merge documents (if applicable).

Launch the Streamlit app (streamlit_frontend_tool.py).

Use the web interface to view, select, or download PDF files.

Modify scripts for your own use case — such as adding upload, filtering, or analysis options.

⚙️ Configuration
Update paths in scripts (merge.py, streamlit_frontend_tool.py) if you change folder names.

Add new dependencies to requirements.txt.

To deploy publicly, use:

Streamlit Community Cloud

or any service that supports Python web apps.

If you use .env files (for API keys, credentials, etc.), you can load them using the python-dotenv package.
