FROM python:3.10-slim-buster

# Install system libraries needed for NumPy and related packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

WORKDIR /app

# Copy requirements and install them
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# (Optional) Debug: Print numpy version to verify installation
RUN python -c "import numpy; print('numpy version:', numpy.__version__)"
RUN python -c "import pandas; print('pandas  version:', pandas.__version__)"
RUN python -c "import sklearn; print('sklearn version:', sklearn.__version__)"

# Copy necessary files

COPY download_model.py .
COPY requirements.txt .
COPY ./static ./static
COPY app.py .

EXPOSE 5000

CMD ["sh", "-c", "python download_model.py && python app.py"]
