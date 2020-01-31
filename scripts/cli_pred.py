from model import MatchModel

MODEL_DIR = "model"
model = MatchModel.load(MODEL_DIR)

while True:
    text = input("Input sentence: ")
    a, b, c, *_ = text.split()
    results = model.predict([(a, b, c)])

    for label, prob in results:
        print(str(label), prob)
