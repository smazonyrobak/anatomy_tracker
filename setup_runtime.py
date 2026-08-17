from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
BASE_URL = "https://github.com/smazonyrobak/anatomy_tracker/releases/download/runtime-assets-v1"
ASSETS = {
    "models/DeepSlice/deepslice_mouse_primary_opset18.onnx": "90ce8d4662f53a602035a99d5145c0e6ae8924cde7f9de440cf6b74f79c791ac",
    "models/DeepSlice/deepslice_mouse_secondary_opset18.onnx": "2d7b5e44d9dc4aa6009df6c3cc7e8a0cbb9fd33dc63a8bd2ac43ea5999237978",
    "models/AtlasPose/atlas_pose.onnx": "803383bde833bf1cb9549c2a9f44314c5c7827d2f67527901ec7a76f5d7a6495",
    "models/DiffeomorphicRegistration/dense_registration.onnx": "567076b75039ee5f6498918feedfe638237ff4be067dfadc915a40d5c36b8dce",
    "data/Allen Brain Atlas 25um/average_template_25.nrrd": "e4a2b483e842b4c8c1b5452d940ea59e14bc1ebaa38fe6a9c3bacac6db2a8f4b",
    "data/Allen Brain Atlas 25um/annotation_25.nrrd": "c620cbcc562183e4dcd40250d440130501781f74b41de35b1c1bdabace290c42",
    "data/Allen Brain Atlas 25um/query.csv": "5347daf90e02ac1d1cfcbf9c8af86ff23a2fb32cd7e7a2ba2881951931286dbd",
    "data/Allen Brain Atlas 25um/atlas_meshdata.pkl": "80fb55adbf2bd084f960d562ca4f12bdf1bc3c60aabbcd378ba780ee5562a2e2",
}


def digest(path):
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


for relative_path, expected in ASSETS.items():
    destination = ROOT / relative_path
    if destination.is_file() and digest(destination) == expected:
        print(f"ready: {relative_path}")
        continue
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"downloading: {relative_path}")
    request = Request(f"{BASE_URL}/{destination.name}", headers={"User-Agent": "anatomy-tracker-setup"})
    with urlopen(request) as response, temporary.open("wb") as output:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            output.write(chunk)
    if digest(temporary) != expected:
        temporary.unlink()
        raise RuntimeError(f"SHA-256 mismatch for {relative_path}")
    temporary.replace(destination)

print("Runtime models and Allen CCF assets are installed and checksum-verified.")
