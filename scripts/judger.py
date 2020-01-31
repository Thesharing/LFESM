import sys
import json


def get_score(ground_truth_path, output_path):
    cnt = 0
    f1 = open(ground_truth_path, "r", encoding='utf-8')
    f2 = open(output_path, "r", encoding='utf-8')
    correct = 0
    for line in f1:
        cnt += 1
        true_data = json.loads(line)
        if true_data['label'] == f2.readline().strip():
            correct += 1

    return 1.0 * correct / cnt


if __name__ == "__main__":
    ground_truth_path = "../data/test/test.json"
    output_path = "../data/test/output.txt"
    if len(sys.argv) == 3:
        ground_truth_path = sys.argv[1]
        output_path = sys.argv[2]

    score = get_score(ground_truth_path, output_path)
    print(score)
