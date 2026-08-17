"""Shim package that lets benchmark.py run unmodified against a live pod.

benchmark.py was written for an in-process retriever (`from app.retriever
import search`). This package keeps that import surface but backs it with
HTTP calls to a deployed Voice RAG API, so the numbers it prints are the
deployment's, not a local index's.

Point it at a pod with RAG_BASE_URL (default is the current live pod).
"""
