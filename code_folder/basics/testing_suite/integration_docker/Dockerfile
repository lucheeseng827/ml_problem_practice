FROM python:3.8

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY src/ /app

# Copy test code
COPY tests/ /tests

# Run tests
CMD ["python", "-m", "unittest", "discover", "-s", "/tests"]
