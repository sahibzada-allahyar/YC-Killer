def ensure_matplotlib_agg():
    import matplotlib
    try:
        matplotlib.get_backend()
    except Exception:
        matplotlib.use("Agg")
