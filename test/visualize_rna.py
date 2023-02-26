import matplotlib.pyplot as plt
from ipynb.draw import draw_struct
import numpy as np

seq    = 'GGAUACGGCCAUACUGCGCAGAAAGCACCGCUUCCCAUCCGAACAGCGAAGUUAAGCUGCGCCAGGCGGUGUUAGUACUGGGGUGGGCGACCACCCGGGAAUCCACCGUGCCGUAUCCU'
struct = '(((((((((....((((((((.....((((((............))))..))....)))))).)).(((((......(((((.(((....)))))))).....))))).))))))))).'

draw_struct(seq, struct)