# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 10:22:35 2018

@author: Daniel Lee
"""
from tkinter import *
from GUI import MyGUI
import predictor
from Graph import Graph

def main():
    graph = Graph()
    graph.saveGraph()
    root = Tk()
    probabilityIncrease, probabilityDecrease = predictor.predict()
    gui = MyGUI(probabilityIncrease, probabilityDecrease, root)
    gui.appOpen()
    root.mainloop()
    
main()
