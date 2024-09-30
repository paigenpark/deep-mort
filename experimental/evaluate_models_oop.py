from pipeline import *
import os

here = os.path.dirname(os.path.abspath(__file__))
paths = [os.path.join(here, "../data/usmdb/usmdb.csv"), os.path.join(here, "../data/hmd.csv")]
pipeline = Pipeline(paths=paths)