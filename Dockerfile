FROM python:3.11.0-slim-buster as python-base
# RUN echo 'Acquire::http::Proxy "socks5h://192.168.1.214:9995";' > /etc/apt/apt.conf.d/proxy.conf && \
#         echo 'Acquire::https::Proxy "socks5h://192.168.1.214:9995";' >> /etc/apt/apt.conf.d/proxy.conf
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME="/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/magni" \
    VENV_PATH="/magni/.venv"
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

FROM python-base as builder-base
ENV POETRY_VERSION=1.2.2
RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl -sSL https://install.python-poetry.org | python
# RUN curl -socks5 socks5h://192.168.1.214:9990 -sSL https://install.python-poetry.org | python
WORKDIR $PYSETUP_PATH
COPY ./pyproject.toml ./
COPY ./poetry.lock ./
# RUN HTTPS_PROXY=http://192.168.1.214:8118 poetry install --without dev
RUN poetry install --without dev

FROM python-base as production
COPY --from=builder-base $VENV_PATH $VENV_PATH
COPY ./magni.py $PYSETUP_PATH/magni.py
COPY ./template.jinja2 $PYSETUP_PATH/
WORKDIR $PYSETUP_PATH
ENTRYPOINT ["/magni/magni.py"]
