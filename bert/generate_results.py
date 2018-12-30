test_results = []
with open("test_results.txt", "r") as f:
    for line in f:
        test_results.append(line.split("\t")[5].strip("\n"))

qid_list = []
with open("data/eval1_unlabelled.tsv", "r") as f:
    for line in f:
        qid = line.split("\t")[0]
        qid_list.append(qid)

assert len(qid_list) == len(test_results)

from collections import defaultdict

score = defaultdict(list)

for i in range(len(qid_list)):
    score[qid_list[i]].append(test_results[i])

for k in score:
    if len(score[k]) != 10:
        print("ALERT: Some Issue")
        print(k, score[k])
        break

with open("answer.tsv","w") as f:
    for k in score:
        f.write(k + "\t" + "\t".join(score[k]) + "\n")