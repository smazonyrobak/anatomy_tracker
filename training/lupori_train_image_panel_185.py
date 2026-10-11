"""Display only frozen TRAIN-animal acquired thumbnails."""

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageDraw

archive = Path('I:/AnatomyTracker/data/lupori_joint_185/rawData.zip')
animals = ['AL1A', 'CC2B', 'CC4B', 'CC4A']
assert archive.stat().st_size == 48_652_184_187
panel = Image.new('RGB', (1280, 1360), 'white')
draw = ImageDraw.Draw(panel)
with ZipFile(archive) as source:
    for i, animal in enumerate(animals):
        names = sorted(n for n in source.namelist() if n.startswith(f'{animal}/thumbnails/') and n.endswith('-thumb.png'))
        name = names[len(names) // 2]
        image = Image.open(BytesIO(source.read(name))).convert('RGB')
        image.thumbnail((620, 620))
        x, y = (i % 2) * 640 + (640 - image.width) // 2, (i // 2) * 680 + 30 + (620 - image.height) // 2
        panel.paste(image, (x, y))
        draw.text(((i % 2) * 640 + 10, (i // 2) * 680 + 5), name.split('/')[-1], fill='black')

output = archive.with_name('train_thumbnail_panel.png')
panel.save(output)
print(output)
