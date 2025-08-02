
# How to Run the Program

## 1. Download Python 3.10.5 Menggunakan Link Di Bawah Dan Sesuaikan Dengan Device

- [Python 3.10.5 for x64](https://www.python.org/ftp/python/3.10.5/python-3.10.5-amd64.exe)
- [Python 3.10.6 for x32](https://www.python.org/ftp/python/3.10.6/python-3.10.6.exe)

pip install mediapipe opencv-python

## 2. Buat dan Aktifkan Virtual Environment (Opsional tapi Direkomendasikan)

Buka Terminal dan jalankan perintah berikut untuk membuat virtual environment:

```bash
python -m venv detection_venv
```

Aktifkan virtual environment:

- **Windows (Command Prompt):**
  ```cmd
  detection_venv\Scripts\activate.bat
  ```
- **Windows (PowerShell):**
  ```powershell
  .\detection_venv\Scripts\Activate.ps1
  ```
- **Mac/Linux:**
  ```bash
  source detection_venv/bin/activate
  ```

## 3. Install Library / Module Yang Dibutuhkan

Pastikan virtual environment sudah aktif, lalu jalankan perintah berikut:

```bash
pip install mediapipe opencv-python
```


## 4. Jalankan Program Python

Jalankan program menggunakan perintah berikut:

```bash
python main.py
```