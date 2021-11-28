from engine.trainer import Trainer
from utils import load_config


cfg_file = "config.yaml"
cfg = load_config(cfg_file)

trainer = Trainer(cfg_file)

steps_per_epoch = len(trainer.train_loader)
total_steps = cfg.PARAMS.EPOCH * steps_per_epoch
start_step = trainer.summary["step"]

for step in range(start_step, total_steps):
    trainer.train_step()

    if (step + 1) % cfg.LOGGING.LOG_EVERY == 0:
        trainer.log(mode="train")

    if (step + 1) % cfg.LOGGING.SAVE_EVERY == 0:
        trainer.dev_epoch()
        trainer.log(mode="dev")
        trainer.save(mode="train")
        trainer.save(mode="dev")

trainer.finish()
