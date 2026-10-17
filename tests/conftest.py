def pytest_configure(config):
    config.addinivalue_line("markers", "workflow: runs the B0/B1 workflow scripts in subprocesses (slow); deselect with -m 'not workflow'")
