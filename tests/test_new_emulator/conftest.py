# content of conftest.py
import pytest
def pytest_addoption(parser):
    parser.addoption("--cmdopt1", action="store", help="Name of an emulator")

