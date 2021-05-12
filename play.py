from tqdm import tqdm
import time
import random

for i in tqdm(range(100), smoothing=0):
    time.sleep(random.random())
