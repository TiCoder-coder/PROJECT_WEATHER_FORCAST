import json

input_path = 'data/data_crawl/Bao_cao_20260211_173719.schema.jsonl'
output_path = 'data/data_crawl/Bao_cao_20260211_173719.schema.json'

with open(input_path, 'r', encoding='utf-8') as fin:
    lines = [json.loads(line) for line in fin if line.strip()]

with open(output_path, 'w', encoding='utf-8') as fout:
    json.dump(lines, fout, ensure_ascii=False, indent=2)

print(f"✅ Đã chuyển {input_path} thành {output_path} (dạng JSON array)")
