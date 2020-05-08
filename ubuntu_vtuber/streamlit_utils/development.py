import numpy as np
import json

import argparse
import streamlit as st

import os
import glob

import time
from collections import defaultdict
import plotly.express as px

import plotly.graph_objects as go

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def compare_function_performances(funcs, args):
    process_times = defaultdict(list)
    
    for func in funcs:
        for arg in args:
            start = time.time()
            result = func(*arg)
            end = time.time()
            process_times[func.__name__].append(dict(
                args=args,
                time=end - start)
            )

    fig = go.Figure()
    for func_name, times in process_times.items():
        fig.add_trace(go.Box(y=[d['time'] * 1000 for d in times], name=func_name))
    st.write(fig)

def write_as_json(obj):
    json_str = json.dumps(obj, indent=4, cls = MyEncoder)
    st.markdown(f"```json\n{json_str}\n```")

def test_output(func, arg):
    st.markdown(f"## {func.__name__}")
    write_as_json(func(*arg))