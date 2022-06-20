#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun 19 Jun 2022 09:57:18 PM CEST

@author hielke
"""

from enum import Enum


class ModelName(str, Enum):
    bow = "bow"
    tfidf = "tfidf"
    mlb = "mlb"
