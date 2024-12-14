import json

total_shard = 10

shard_files = [f"outputs/our_ds-lookback-decoding.jsonl_{i}.jsonl" for i in range(total_shard)]

output_file = "outputs/our_ds-lookback-decoding_combined.json"

combined = {}

for shard_file in shard_files:
    with open(shard_file, "r", encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            combined.update(data)

with open(output_file, "w", encoding='utf-8') as f:
    json.dump(combined, f, ensure_ascii=False, indent=4)
