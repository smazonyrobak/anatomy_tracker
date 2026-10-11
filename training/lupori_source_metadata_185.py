"""Read specimen identities only, before opening any Lupori section image."""

import json
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

archive = Path('I:/AnatomyTracker/data/lupori_joint_185/rawData.zip')
assert archive.stat().st_size == 48_652_184_187
records = []
with ZipFile(archive) as source:
    names = sorted(n for n in source.namelist() if n.endswith('-info.xml'))
    assert len(names) == 7
    for name in names:
        root = ElementTree.fromstring(source.read(name))
        sections = root.findall('slices')
        records.append({'animal_id': root.findtext('mouseID'), 'source_info_xml': name,
            'treatment': root.findtext('treatment'), 'genotype': root.findtext('genotype'),
            'sex': root.findtext('sex'), 'age_days': int(root.findtext('age')),
            'channels': [node.text for node in root.findall('channelNames')],
            'sections': len(sections),
            'valid_sections': sum(node.findtext('valid') == '1' for node in sections),
            'flipped_sections': sum(node.findtext('flipped') == '1' for node in sections)})

output = archive.with_name('source_metadata.json')
output.write_text(json.dumps(records, indent=2), encoding='utf-8')
print(json.dumps(records, indent=2))
