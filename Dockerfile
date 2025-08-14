FROM python:3.10-slim

WORKDIR /app

# Systemabhängigkeiten
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt kopieren und installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# GitHub-Paket installieren
RUN pip install --no-cache-dir git+https://github.com/tqsd/special_issue_quantum.git@master

# Symlink nur setzen, wenn qsi noch nicht existiert
RUN python -c "import site, os; site_dir = site.getsitepackages()[0]; qsi = os.path.join(site_dir, 'qsi'); target = os.path.join(site_dir, 'special_issue_quantum'); (not os.path.exists(qsi)) and os.symlink(target, qsi)"

# Code kopieren
COPY . /app

CMD ["python","-u", "main.py"]
