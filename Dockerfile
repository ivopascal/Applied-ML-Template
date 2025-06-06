FROM python:3.12 AS builder

WORKDIR /usr/src/app
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN pip install --upgrade pip
COPY . .
RUN pip install --no-cache-dir -r requirements.txt



FROM python:3.12 AS service

WORKDIR /root/app/site-packages
COPY --from=builder /venv /venv
COPY . .
ENV PATH=/venv/bin:$PATH
