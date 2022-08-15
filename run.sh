TASK_NAME_LIST=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "stsb" "wnli")

export NUM_EPOCHS=3

#export ppl_dataset="None"
#export ppl_dataset="ADReSS20-train-transcript"
#export ppl_dataset="ADReSS20-train-transcript-HC-icu"
#export ppl_dataset="ADReSS20-train-transcript-AD-icu"
#export ppl_dataset="ADReSS20-train-transcript-diff-icu"
export ppl_dataset="ADReSS20-train-transcript-rdiff-icu"
#export ppl_dataset="ADReSS20-train-transcript-HC-AD-icu"

#export TASK_NAME=mrpc
for TASK_NAME in ${TASK_NAME_LIST[*]}; do
   echo $TASK_NAME

  #CUDA_VISIBLE_DEVICES=7 python run_glue.py \
  #  --model_name_or_path bert-base-uncased \
  #  --task_name $TASK_NAME \
  #  --do_train \
  #  --do_eval \
  #  --max_seq_length 128 \
  #  --per_device_train_batch_size 32 \
  #  --learning_rate 2e-5 \
  #  --num_train_epochs $NUM_EPOCHS \
  #  --output_dir checkpoints/$TASK_NAME/

  CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch --config_file distribute_accelerate_config.yaml run_glue_no_trainer.py \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --max_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs $NUM_EPOCHS \
    --ppl_dataset $ppl_dataset \
    --output_dir checkpoints_ppl/$ppl_dataset/$NUM_EPOCHS/$TASK_NAME/

#  CUDA_VISIBLE_DEVICES=0 python run_glue_no_trainer.py \
#    --model_name_or_path bert-base-uncased \
#    --task_name $TASK_NAME \
#    --max_length 128 \
#    --per_device_train_batch_size 32 \
#    --learning_rate 2e-5 \
#    --num_train_epochs $NUM_EPOCHS \
#    --ppl_dataset $ppl_dataset \
#    --output_dir checkpoints_ppl/$ppl_dataset/$NUM_EPOCHS/$TASK_NAME/
done