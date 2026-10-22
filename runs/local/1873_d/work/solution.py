from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import List, Dict, Set, Tuple, Optional, Union, Any
import string, re, datetime, collections, heapq, bisect, copy, math, random, statistics, itertools, functools, operator, io, sys, json
sys.setrecursionlimit(50000)

import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    t = int(data[idx]); idx += 1
    out = []
    for _ in range(t):
        n = int(data[idx]); k = int(data[idx + 1]); idx += 2
        s = data[idx]; idx += 1
        count = 0
        covered_until = 0
        for i in range(n):
            if s[i] == 'B' and i > covered_until:
                count += 1
                covered_until = i + k
        out.append(str(count))
    sys.stdout.write("\n".join(out) + "\n")

main()
