NICKNAME="cpdino_300_fp32_full"

python cp_posh/ssl/main/main_dino.py \
wandb.project="cp_posh" \
nickname="${NICKNAME}" \
trainer.num_nodes=2 \
trainer.devices=8 \
trainer.precision=32 \
trainer.max_epochs=100 \
trainer.default_root_dir="./checkpoints/${NICKNAME}" \
meta_arch/backbone=vit_small \
data=cp_posh_300 \
data.cp_posh_300.args.channels=[0,1,2,3,4] \
data.cp_posh_300.loader.num_workers=32 \
data.cp_posh_300.loader.batch_size=28 \
data.cp_posh_300.loader.drop_last=True \
transformations=cp_posh_300