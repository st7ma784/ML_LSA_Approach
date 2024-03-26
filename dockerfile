# Use the pytorch base image
FROM pytorch/pytorch:latest
RUN apt-get -y update
RUN apt-get -y install git

# Copy the requirements.txt file to the container
COPY requirements.txt /app/requirements.txt

# Install the Python dependencies
RUN pip install -r /app/requirements.txt

# Set the working directory
WORKDIR /app

# Copy the rest of the files to the container
COPY . /app

# Run the launch command with the num_trials -1 flag

#use ENTRYPOINT to run the command
CMD ["python", "launch.py", "--dir","/data", "--num_trials","-1"]