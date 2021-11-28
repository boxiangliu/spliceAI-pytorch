import wandb

wandb.login()
wandb.init(project="test",name="run1", save_code=True)
wandb.log({"loss":5}, step=2)
wandb.log({"acc":6}, step=4)
wandb.finish()
