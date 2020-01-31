import logging
import sys
import time

import torch

from model import MatchModel
from data import TripletTextDataset
from util import seed_all

logging.disable(sys.maxsize)

start_time = time.time()
input_path = "./data/test/test.json"
output_path = "./data/test/output.txt"

if len(sys.argv) == 3:
    input_path = sys.argv[1]
    output_path = sys.argv[2]

inf = open(input_path, "r", encoding="utf-8")
ouf = open(output_path, "w", encoding="utf-8")

seed_all(42)

MODEL_DIR = "./output/model"
model = MatchModel.load(MODEL_DIR, torch.device("cpu"))
print('Model: ' + MODEL_DIR)

test_set = TripletTextDataset.from_jsons(input_path)

results = model.predict(test_set)

for label, _ in results:
    print(str(label), file=ouf)

inf.close()
ouf.close()

end_time = time.time()
spent = end_time - start_time
print("numbers of samples: %d" % len(results))
print("time spent: %.2f seconds" % spent)
