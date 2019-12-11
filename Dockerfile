FROM pytorch/pytorch:latest
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
CMD ["/bin/bash"]