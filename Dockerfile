# 1. Use the official Python 3.11 image
FROM python:3.11-slim

# 2. Set the working directory inside the server
WORKDIR /app

# 3. Install the missing C++ libraries that MediaPipe needs
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy your requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your app code
COPY . .

# 6. Expose the port so Vercel can talk to it
EXPOSE 8000

# 7. Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]