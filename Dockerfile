FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir .
COPY ui.py ./
COPY examples ./examples
RUN useradd --create-home appuser && mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser
EXPOSE 8501
CMD ["streamlit", "run", "ui.py", "--server.address", "0.0.0.0", "--browser.gatherUsageStats", "false"]
