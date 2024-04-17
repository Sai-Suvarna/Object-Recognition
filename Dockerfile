FROM python:3.11
WORKDIR /code
COPY . /code
RUN pip install -r requirements.txt
RUN pip install -r requirements1.txt
RUN pip install -r requirements2.txt
RUN pip install -r requirements3.txt
CMD ["uvicorn", "app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
