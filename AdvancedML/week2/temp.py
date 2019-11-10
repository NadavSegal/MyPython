#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:21:11 2019

@author: nadav
"""

import sys
sys.path.append("..")
import tqdm_utils
import download_utils


# In[ ]:


# use the preloaded keras datasets and models
download_utils.link_all_keras_resources()