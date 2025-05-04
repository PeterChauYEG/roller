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
python -m src.agent.trainer --timesteps 10000 \
--batch_size 128 \
--save_checkpoint_frequency 200000 \
--linear_lr_schedule
--experiment_name lower_player_damage
--save_model_path lower_player_damage_model.zip
--save_model_path model.zip
```

## inference
```
python -m src.agent.inference --timesteps 1000 \
--render
--model_path model.zip
--model_path --model_path model.zip
```
