

import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

def append_items_to_file(filename, items):
    with open(filename, 'a') as f:
        for item in items:
            f.write(str(item) + ' ')
        f.write('\n')

def write_items_to_file(filename, items):
    with open(filename, 'w') as f:
        for item in items:
            f.write(str(item) + '\n')

def read_file_to_items(filename):
    items = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items.append(line)

    return items