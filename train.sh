: "${TARGET:=EuroSAT}"
: "${SOURCE:=miniImageNet}"
: "${BACKBONE:=resnet10}"

python main.py system=ce_distill_ema_sgd trainer.gpus=1 backbone=$BACKBONE \
  source_data=$SOURCE trainer.max_epochs=60 \
  data.val_dataset=${TARGET}_test data.test_dataset=null print_val=false \
  trainer.log_every_n_steps=-1 \
  unlabel_params.dataset=${TARGET}_train data.num_episodes=600 \
  trainer.progress_bar_refresh_rate=0 print_val=false launcher.gpus=1 \
  model_name=dynamic_cdfsl_${SOURCE}_${TARGET}
