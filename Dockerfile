# Base image - Tensorflow 2.x support-kaga Python 3.10 use pandrom
FROM python:3.10

# Hugging Face Spaces-la security-kaga root user allow panna mattanga. 
# Adhunala 'user' nu oru puthu user (ID 1000) create pandrom.
RUN useradd -m -u 1000 user
USER user

# Environment variables set pandrom
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Working directory set pandrom
WORKDIR $HOME/app

# Unga GitHub files ellam container-ku copy pandrom
COPY --chown=user . $HOME/app

# requirements.txt-la iruka libraries ellam install pandrom
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face default port 7860-a expose pandrom
EXPOSE 7860

# App-a run pandrom
CMD ["python", "app.py"]
