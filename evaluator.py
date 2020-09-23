import argparse
import json

"""
Example Usage: 
python evaluator.py eval --test_file datasets/MCTACO/test_9442.tsv --prediction_file models/others/bert.norm.output.txt
python evaluator.py eval --test_file datasets/MCTACO/test_9442.tsv --prediction_file models/others/roberta.output.txt
python evaluator.py eval --test_file datasets/MCTACO/test_9442.tsv --prediction_file models/roberta-base/pred_results_roberta-base.txt
python evaluator.py eval --test_file datasets/MCTACO/test_9442.tsv --prediction_file models/roberta-large/pred_results_roberta-large.txt
python evaluator.py eval --test_file datasets/MCTACO/test_9442.tsv --prediction_file models/multi-mctaco-ner-roberta-large/pred_results_roberta-large.txt
python evaluator.py eval --test_file datasets/MCTACO/test_9442.tsv --prediction_file models/bert-base-cased/pred_results_bert-base-cased.txt
python evaluator.py eval --test_file datasets/MCTACO/test_9442.tsv --prediction_file models/bert-base-uncased/pred_results_bert-base-uncased.txt
python evaluator.py eval --test_file datasets/MCTACO/test_9442.tsv --prediction_file models/bert-large-uncased/pred_results_bert-large-uncased.txt
python evaluator.py eval --test_file datasets/MCTACO/test_9442.tsv --prediction_file models/ner-from-tbaq-class-event-bert-base-uncased/pred_results_bert-base-uncased.txt
python evaluator.py eval --test_file datasets/MCTACO/test_9442.tsv --prediction_file models/ner-from-tbaq-class-timex-bert-base-uncased/pred_results_bert-base-uncased.txt
python evaluator.py eval --test_file datasets/MCTACO/test_9442.tsv --prediction_file models/bert-base-uncased-mctaco/eval_outputs.txt
python evaluator.py eval --test_file datasets/MCTACO/test_9442.tsv --prediction_file models/sbert-bert-base-uncased/pred_results_sbert_results.txt
python evaluator.py eval --test_file datasets/MCTACO/test_9442.tsv --prediction_file models/sbert-roberta-base/pred_results_sbert_results.txt
python evaluator.py eval --test_file datasets/MCTACO/test_9442.tsv --prediction_file models/fairseq-roberta-large/preds.txt
python evaluator.py eval --test_file datasets/MCTACO/ppt_data.tsv --prediction_file models/others/ppt_data.txt
python evaluator.py eval --test_file datasets/MCTACO/dev_splitted.tsv --prediction_file models/multi-mctaco-ner-roberta-large/pred_results_roberta-large.txt
python evaluator.py eval --test_file datasets/MCTACO/test_9442.tsv --prediction_file multi_results/1/pred_results_roberta-large.txt
python evaluator.py eval --test_file datasets/MCTACO/dev_splitted_sent.tsv --prediction_file multi_results_2e-5/13_dev/pred_results.txt
python evaluator.py eval --test_file datasets/MCTACO/test_9442.tsv --prediction_file multi_results/test/pred_results.txt
"""


class McTacoEvaluator:

    def __init__(self, test_file, output_file, output):
        self.test_file = test_file
        self.output_file = output_file
        self.output = output

    def print_result(self):
        ref_lines = [x.strip() for x in open(self.test_file).readlines()]
        prediction_lines = [x.strip() for x in open(self.output_file).readlines()]

        result_map = {}
        prediction_count_map = {}
        prediction_map = {}
        gold_count_map = {}
        for i, line in enumerate(ref_lines):
            key = " ".join(line.split("\t")[0:2])
            if key not in result_map:
                result_map[key] = []
                prediction_count_map[key] = 0.0
                gold_count_map[key] = 0.0
                prediction_map[key] = []
            prediction = prediction_lines[i]
            prediction_map[key].append(prediction)
            label = line.split("\t")[3]
            if prediction == "yes":
                prediction_count_map[key] += 1.0
            if label == "yes":
                gold_count_map[key] += 1.0
            result_map[key].append(prediction == label)

        total = 0.0
        correct = 0.0
        f1 = 0.0
        for question in result_map:
            val = True
            total += 1.0
            cur_correct = 0.0
            for i, v in enumerate(result_map[question]):
                val = val and v
                if v and prediction_map[question][i] == "yes":
                    cur_correct += 1.0
            if val:
                correct += 1.0
            p = 1.0
            if prediction_count_map[question] > 0.0:
                p = cur_correct / prediction_count_map[question]
            r = 1.0
            if gold_count_map[question] > 0.0:
                r = cur_correct / gold_count_map[question]
            if p + r > 0.0:
                f1 += 2 * p * r / (p + r)

        print("Correct: " + str(correct))
        print("F1: " + str(f1))
        print("Total: " + str(total))
        print("Strict Acc.: {:.3f}" .format(correct / total))
        print("Avg F1: {:.3f}" .format(f1 / total))

        if self.output:
            print("Writing results to file: %s" % self.output)
            with open(self.output, "wt", encoding="UTF-8") as output:
                output.write(json.dumps({
                    "em": correct / total,
                    "f1": f1 / total
                }))

    def print_errors(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="[eval]")
    parser.add_argument("--test_file",
                        required=True,
                        help="path to the csv file with gold labels.")
    parser.add_argument("--prediction_file",
                        required=True,
                        help="path to the line-by-line file containing system predictions.")
    parser.add_argument(
        '--output', '-o',
        help='Output results to this file.')

    args = parser.parse_args()
    if args.command == "eval":
        evaluator = McTacoEvaluator(args.test_file, args.prediction_file, args.output)
        evaluator.print_result()
    else:
        print("Command not found, see --help.")


if __name__ == "__main__":
    main()