# RingPong

## setup
### environment setup

```bash
python3 -m venv venv
source venv/bin/activate
```

### export packages
```bash
pip install pipreqs
pipreqs . --force
```

### install packages
```bash
pip install -r requirements.txt
```

--------------

## train
```
python -m src.agent.trainer
```

## inference
```
python -m src.agent.inference
```
