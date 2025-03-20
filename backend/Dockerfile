FROM python:3.9

WORKDIR /app

COPY . .

RUN python -m venv venv

RUN . venv/bin/activate && pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["sh", "-c", ". venv/bin/activate && python app.py"]