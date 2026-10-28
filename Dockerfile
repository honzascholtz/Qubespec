# QubeSpec container: Python + QubeSpec + JupyterLab for the tutorials.
#   docker build -t qubespec .
#   docker run --rm -p 8888:8888 -v "$PWD":/work qubespec
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

# Compilers are needed for packages without wheels (e.g. sep, spectres on some archs)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/qubespec

# Editable install so the full source tree (data files, Voronoi, ...) is used,
# plus optional dependencies used by some submodules
COPY setup.py README.md MANIFEST.in ./
COPY QubeSpec ./QubeSpec
RUN pip install --upgrade pip \
    && pip install -e . \
    && pip install \
        networkx \
        pyneb \
        regions \
        vorbin \
        pyyaml \
        sedpy \
        jupyterlab

# Sanity check that the package imports
RUN python -c "import QubeSpec; print('QubeSpec imported OK')"

WORKDIR /work
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=", "--ServerApp.password="]
