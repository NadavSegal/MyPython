# https://github.com/hse-aml/intro-to-dl/blob/master/README.md
######### my additions:
Out = 'C:/Users/Nadav/Documents/MyPython/AdvancedML/week1'
import sys
sys.path.append(Out)
import os
os.chdir(Out)

import wget
wget.download('https://raw.githubusercontent.com/hse-aml/intro-to-dl/master/setup_google_colab.py',\
              out = Out)
######### end my additions 
# for using google colab:
import setup_google_colab

#setup_google_colab.setup_week1
#setup_google_colab.setup_week2
#setup_google_colab.setup_week3
#setup_google_colab.setup_week4
#setup_google_colab.setup_week5
#setup_google_colab.setup_week6