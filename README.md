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

## lint 
```bash
flake8 src; black src
```

--------------

## train
```
python -m src.agent.trainer --timesteps 1000000 \
--batch_size 512 \
--save_checkpoint_frequency 100000 \
--linear_lr_schedule \
--experiment_name lower_player_damage \
--save_model_path model.zip
```

## inference
```
python -m src.agent.inference --timesteps 10 \
--render \
--model_path model.zip

python -m src.agent.inference --timesteps 1000 \
--model_path model.zip
```
