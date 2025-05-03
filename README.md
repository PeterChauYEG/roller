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
python -m src.agent.trainer --timesteps 100 \
--batch_size 32 \
--save_checkpoint_frequency 200000 \
--linear_lr_schedule

```

## inference
```
python -m src.agent.inference --timesteps 1 \
--render
```
