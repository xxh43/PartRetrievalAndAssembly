
import numpy as np

class Shape:
    def __init__(self):
        self.parts = []
        self.part_start_index = None
        self.pc = None
        self.pc_normals = None
        self.mesh = None

class Part:
    def __init__(self):
        self.pc = None
        self.pc_normals = None
        self.mesh = None
