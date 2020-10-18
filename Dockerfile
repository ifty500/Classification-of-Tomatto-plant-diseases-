FROM python:3.7.7
COPY . /app
EXPOSE 7000
WORKDIR /app
RUN pip install -r requirements.txt
CMD python app.py