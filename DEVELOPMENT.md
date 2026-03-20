## Building
```bash
python -m pip install --upgrade build
python -m build

```

This will create dist/** files.

For choosing one type or the other:
```bash

python -m build --wheel   # wheel only
python -m build --sdist   # sdist only
```

Installing on isolated environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install dist/imagebreed_client-0.1.0-py3-none-any.whl
```


