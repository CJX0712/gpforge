FROM python:3.11-slim

LABEL author="晨星" \
      description="gpforge — deterministic Gaussian Process toolkit"

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY README.md ./
COPY gpforge/ ./gpforge/
COPY examples/ ./examples/
COPY tests/ ./tests/

ENV PYTHONPATH=/app
ENV PYTHONHASHSEED=0

# Install the package itself (editable-free, plain copy install)
RUN pip install --no-cache-dir --no-deps .

ENTRYPOINT ["python", "-m", "gpforge.cli"]
CMD ["methods"]
