FROM python:3.12 AS builder

WORKDIR /workspace
COPY benchmark benchmark
COPY setup.py .

RUN pip install setuptools -i http://mirrors.sangfor.org/pypi/simple --trusted-host mirrors.sangfor.org
RUN python setup.py bdist_wheel


FROM python:3.12

WORKDIR /workspace

COPY dataset /workspace/dataset
COPY --from=builder /workspace/dist /workspace/dist

RUN pip install --no-cache-dir /workspace/dist/*.whl -i http://mirrors.sangfor.org/pypi/simple --trusted-host mirrors.sangfor.org

RUN rm -rf /workspace/dist
