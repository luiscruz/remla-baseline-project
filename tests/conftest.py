#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 01:07:50 PM CEST 2022

This file is for fixtures. Pytest will automatically "discover" fixtures in
any file named `conftest.py` and there are more ways to "discover" fixtures.

@author hielke
"""
import pytest

from pathlib import Path


@pytest.fixture()
def root_folder():
    cwd = Path().absolute()
    if 'test' in cwd.parts[-1]:
        # run inside the tests folder
        return cwd.parent
    return cwd


@pytest.fixture()
def data_folder(root_folder):
    yield root_folder / 'data'


@pytest.fixture()
def test_folder(root_folder):
    yield root_folder / 'tests'
