FROM wisrovi/open_alpr:1.0

# Install in-memory high-performance API server using precompiled C++ OpenALPR bindings
COPY api.py /data/api.py

WORKDIR /data
EXPOSE 8000

CMD ["python3", "api.py"]
