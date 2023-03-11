FROM python:3.10.2

MAINTAINER calgo-lab

# Pre-installed some packages
COPY pyproject.toml poetry.lock /demo_search/

WORKDIR /demo_search

RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root --no-interaction --no-ansi

COPY demo /demo_search/demo

EXPOSE 8501

COPY /demo .

ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]