from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")

    #--lora_enable True --lora_r 32 --lora_alpha 48 --lora_dropout 0.05 --mm_projector_lr 2e-5 --model_name_or_path  /datasets/work/vLLM/FFAA-master/checkpoints/ffaa-mistral-7b --version v1 --data_path /datasets/newout/eFFAA_dataset.json --image_folder /datasets/newout --vision_tower /datasets/work/vLLM/FFAA-master/models/clip-vit-large-patch14-336 --mm_projector_type mlp2x_gelu --mm_vision_select_layer -2 --mm_use_im_start_end False --mm_use_im_patch_token False --image_aspect_ratio pad --group_by_modality_length True --bf16 True --output_dir /datasets/work/vLLM/FFAA-master/checkpoints/effaa-llava-mistral-7b-lora_test --num_train_epochs 3 --per_device_train_batch_size 8 --per_device_eval_batch_size 4 --gradient_accumulation_steps 1 --save_strategy "steps" --save_steps 5000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --model_max_length 2048 --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --report_to "none"
