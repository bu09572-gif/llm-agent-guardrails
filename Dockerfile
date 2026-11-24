# Approval inbox + MCP filing server. The filings are downloaded and indexed on first start.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src:/app
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY agent.py data.py run_eval.py filing_negatives.txt ./
RUN useradd --create-home --uid 1000 app && mkdir -p data && chown -R app /app
USER app
EXPOSE 8100
CMD ["sh", "-c", "[ -f data/gold/financebench/chunks.parquet ] || python -m docpipe.run --scope eval; uvicorn toolguard.app:app --host 0.0.0.0 --port 8100"]
