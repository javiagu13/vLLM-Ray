# Use the official Python image as base
FROM python:3.10.12

# Set working directory
WORKDIR /app

# Install required packages
RUN pip install vllm lm-format-enforcer pandas ray[serve] ipython

# Copy your Python file into the container
COPY vllm_ray_inference.py /app/
COPY models /app/models

# Copy the CUDA Toolkit installer from your host machine to the container
COPY cuda/* /app/cuda/

# Make the installer executable
#RUN chmod +x /app/cuda/cuda_12.1.0_530.30.02_linux.run
RUN chmod +x /app/cuda/cuda_12.2.0_535.54.03_linux.run 


# Install the CUDA Toolkit from the installer, automatically accepting the EULA
RUN /app/cuda/cuda_12.2.0_535.54.03_linux.run --toolkit --silent --toolkitpath=/usr/local/cuda  && \
    rm /app/cuda/cuda_12.2.0_535.54.03_linux.run

# Install Vim
RUN apt-get update && apt-get install -y vim

# Expose the port for the API
EXPOSE 8000

# Command to run your Python file
CMD ["python3", "vllm_ray_inference.py"]
