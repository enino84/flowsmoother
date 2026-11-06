# flowsmoother: EnKS-MC with flow-dictated cross-time precision, on swesphere 0.6.0
FROM python:3.13-slim

# One BLAS thread per process: parallelism is over seeds, runs and members.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
RUN pip install -r requirements.txt
COPY src ./src
RUN pip install --no-deps -e .
COPY experiments ./experiments
COPY figures ./figures
COPY tests ./tests
COPY docs ./docs
COPY run.sh ./
RUN chmod +x run.sh && mkdir -p /app/results && chmod 777 /app/results

VOLUME ["/app/results"]
ENTRYPOINT ["./run.sh"]
CMD ["help"]
