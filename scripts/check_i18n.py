import json, re, pathlib

vi = json.load(open('Weather_Forcast_App/locales/vi.json', encoding='utf-8'))
en = json.load(open('Weather_Forcast_App/locales/en.json', encoding='utf-8'))

def get_nested(obj, path):
    for p in path.split('.'):
        if isinstance(obj, dict):
            obj = obj.get(p)
        else:
            return None
    return obj

keys = set()
for f in pathlib.Path('Weather_Forcast_App/templates').rglob('*.html'):
    content = f.read_text(encoding='utf-8', errors='ignore')
    for m in re.finditer(r'\{%\s*t\s+"([^"]+)"', content):
        keys.add(m.group(1))

# Also check JS files
for f in pathlib.Path('Weather_Forcast_App/static/weather/js').rglob('*.js'):
    content = f.read_text(encoding='utf-8', errors='ignore')
    for m in re.finditer(r'\{%\s*t\s+"([^"]+)"', content):
        keys.add(m.group(1))

missing_vi = sorted([k for k in keys if get_nested(vi, k) is None])
missing_en = sorted([k for k in keys if get_nested(en, k) is None])

print(f'Total keys used in templates: {len(keys)}')
print(f'Missing in vi.json: {len(missing_vi)}')
for k in missing_vi:
    print(f'  {k}')
print(f'Missing in en.json: {len(missing_en)}')
for k in missing_en:
    print(f'  {k}')
