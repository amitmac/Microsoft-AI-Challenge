eval_labels, qids_list = {}, []
save_eval_data_path = "data/evaldata_main_phase_2.tsv"
with open(save_eval_data_path, "r") as f:
    for line in f:
        content = line.split("\t")
        qid, ques, para, label, sn = content[0], content[1], content[2], content[3], content[4].strip("\r\n")
        qids_list.append(qid)
        if qid in eval_labels:
            eval_labels[qid].append(int(label))
        else:
            eval_labels[qid] = [int(label)]

# Test
for qid in eval_labels:
    if len(eval_labels[qid]) != 10:
        print(qid, len(eval_labels[qid]))

test_results = []
with open("models/test_results.txt", "r") as f:
    for line in f:
        test_results.append(float(line.split("\t")[1]))

from collections import defaultdict

score = defaultdict(list)

for i in range(len(qids_list)):
    score[qids_list[i]].append(test_results[i])

rr = []
for qid in score:
    qid_score = sorted(zip(score[qid], eval_labels[qid]), key=lambda x: -x[0])
    for ind, (sc, l) in enumerate(qid_score):
        if l == 1:
            rr.append(1./(ind+1))
            break
mrr = sum(rr)/len(rr)

print("score", mrr)