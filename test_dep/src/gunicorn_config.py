"""Gunicorn config file, used to make Prometheus client work with gunicorn"""
from prometheus_client import multiprocess


def child_exit(server, worker):
    multiprocess.mark_process_dead(worker.pid)
