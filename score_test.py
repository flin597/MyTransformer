result_path = "test/test_result.tsv"

# 每组的总题数和通过数，例如 {"A": [20, 15]}
groups = {}
missing = []

with open(result_path, encoding="utf-8") as f:
    lines = f.read().splitlines()

for line in lines[1:]:  # 第一行是表头
    columns = line.split("\t")
    question_id = columns[0]
    tier = columns[1]
    mark = columns[6].strip()

    if tier not in groups:
        groups[tier] = [0, 0]
    groups[tier][0] += 1

    if mark == "1":
        groups[tier][1] += 1
    elif mark != "0":
        missing.append(question_id)

if missing:
    print("Not graded yet:", missing)

total = 0
passed = 0
for tier in sorted(groups):
    tier_total, tier_passed = groups[tier]
    total += tier_total
    passed += tier_passed
    print(f"Tier {tier}: {tier_passed}/{tier_total}")

print(f"Total: {passed}/{total} = {passed / total:.0%}")
