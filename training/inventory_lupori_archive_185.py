"""Inventory the completed Lupori archive before accessing any specimen image."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

archive = Path('I:/AnatomyTracker/data/lupori_joint_185/rawData.zip')
output = archive.with_name('inventory.jsonl')
assert archive.stat().st_size == 48_652_184_187
with archive.open('rb') as stream:
    assert hashlib.file_digest(stream, 'md5').hexdigest() == 'c8c57aa74182207db67aa0b8072be3a7'

with ZipFile(archive) as zip_file, output.open('w', encoding='utf-8') as stream:
    entries = zip_file.infolist()
    for item in entries:
        stream.write(json.dumps({'name': item.filename, 'bytes': item.file_size,
                                 'compressed_bytes': item.compress_size,
                                 'crc32': item.CRC, 'directory': item.is_dir()}) + '\n')

print(json.dumps({'archive': str(archive), 'entries': len(entries),
                  'prefixes': Counter('/'.join(item.filename.split('/')[:2]) for item in entries).most_common(30),
                  'suffixes': Counter(Path(item.filename).suffix.lower() for item in entries).most_common(),
                  'inventory': str(output)}, indent=2))
