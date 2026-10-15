"""Read only the ZIP directory of the CC0 individual-brain MRI archive."""
import struct
import urllib.request
import zlib


url = 'https://zenodo.org/records/5834507/files/MRI_C57_data.zip?download=1'
with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30) as response:
    total = int(response.headers['Content-Length'])
with urllib.request.urlopen(urllib.request.Request(url, headers={
        'Range': f'bytes={total - 65536}-{total - 1}'}), timeout=30) as response:
    tail = response.read()
end = tail.rfind(b'PK\x05\x06')
assert end >= 0
footer = struct.unpack_from('<4s4H2IH', tail, end)
assert footer[1] == footer[2] == 0 and footer[3] == footer[4]
size, offset = footer[5:7]
if offset == 0xffffffff:
    locator = tail.rfind(b'PK\x06\x07', 0, end)
    assert locator >= 0
    zip64_offset = struct.unpack_from('<4sIQI', tail, locator)[2]
    with urllib.request.urlopen(urllib.request.Request(url, headers={
            'Range': f'bytes={zip64_offset}-{zip64_offset + 55}'}), timeout=30) as response:
        zip64 = struct.unpack('<4sQ2H2I4Q', response.read())
    assert zip64[0] == b'PK\x06\x06' and zip64[7] == footer[4]
    size, offset = zip64[8:10]
print('archive_bytes=', total, 'directory_bytes=', size, 'directory_offset=', offset,
      'entries=', footer[4])
with urllib.request.urlopen(urllib.request.Request(url, headers={
        'Range': f'bytes={offset}-{offset + size - 1}'}), timeout=30) as response:
    directory = response.read()
assert len(directory) == size

position = 0
selected = None
subjects = set()
for _ in range(footer[4]):
    entry = struct.unpack_from('<4s6H3I5H2I', directory, position)
    assert entry[0] == b'PK\x01\x02'
    name = directory[position + 46:position + 46 + entry[10]].decode()
    parts = name.split('/')
    if len(parts) > 3:
        subjects.add(parts[2])
    if name.endswith('/C1/rigidaffine_Lddm_fa.hdr'):
        selected = entry
    position += 46 + entry[10] + entry[11] + entry[12]
assert position == size
assert selected is not None
print('subjects=', sorted(subjects), 'selected_header_compressed_bytes=', selected[8])

with urllib.request.urlopen(urllib.request.Request(url, headers={
        'Range': f'bytes={selected[16]}-{selected[16] + 29}'}), timeout=30) as response:
    local = response.read()
assert local[:4] == b'PK\x03\x04'
name_bytes, extra_bytes = struct.unpack_from('<2H', local, 26)
start = selected[16] + 30 + name_bytes + extra_bytes
with urllib.request.urlopen(urllib.request.Request(url, headers={
        'Range': f'bytes={start}-{start + selected[8] - 1}'}), timeout=30) as response:
    header = zlib.decompress(response.read(), -15)
assert len(header) == 348 and struct.unpack_from('<i', header)[0] == 348
print('C1_FA_header=', {'dimensions': struct.unpack_from('<8h', header, 40),
      'datatype': struct.unpack_from('<h', header, 70)[0],
      'bitpix': struct.unpack_from('<h', header, 72)[0],
      'pixdim': struct.unpack_from('<8f', header, 76),
      'magic': header[344:348].hex(), 'orientation_byte': header[252],
      'originator': header[253:263].hex(),
      'description': header[148:228].split(b'\0')[0].decode(errors='replace')})
