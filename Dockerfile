# Get base image
FROM python:3.9

# Set app directory as working directory
WORKDIR /app

# Install all dependencies
COPY ./requirements.txt /app
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all files to app directory
COPY . /app

# Expose port for api hosting
EXPOSE 8080

# Run the API
CMD ["uvicorn", "app:app", "--reload", "--host", "0.0.0.0", "--port", "8080"]

## NOTE: Steps to run docker container with dockerfile:
#     docker build -t art-critiq-app .
#     docker run --name art-critiq -p 8080:8080 art-critiq-app

