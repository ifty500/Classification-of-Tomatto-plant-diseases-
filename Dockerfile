FROM python:3.8-slim-buster
COPY . /app
EXPOSE 7000
WORKDIR /app
RUN pip install -r requirements.txt
CMD python app.py