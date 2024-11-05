这个预训练模型微调的时候，参数
[--pretrained_model_path PRETRAINED_MODEL_PATH]
[--output_model_path OUTPUT_MODEL_PATH] 
[--vocab_path VOCAB_PATH]
[--spm_model_path SPM_MODEL_PATH] --train_path TRAIN_PATH --dev_path DEV_PATH
[--test_path TEST_PATH] [--config_path CONFIG_PATH]
[--embedding {word,word_pos,word_pos_seg,word_sinusoidalpos}]
[--max_seq_length MAX_SEQ_LENGTH] [--relative_position_embedding]
[--relative_attention_buckets_num RELATIVE_ATTENTION_BUCKETS_NUM]
[--remove_embedding_layernorm] [--remove_attention_scale]
[--encoder {transformer,rnn,lstm,gru,birnn,bilstm,bigru,gatedcnn}]
[--mask {fully_visible,causal,causal_with_prefix}]
[--layernorm_positioning {pre,post}] [--feed_forward {dense,gated}]
[--remove_transformer_bias] [--layernorm {normal,t5}] [--bidirectional]
[--factorized_embedding_parameterization] [--parameter_sharing]
[--learning_rate LEARNING_RATE] [--warmup WARMUP] [--fp16]
[--fp16_opt_level {O0,O1,O2,O3}] [--optimizer {adamw,adafactor}]
[--scheduler {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
[--batch_size BATCH_SIZE] [--seq_length SEQ_LENGTH] [--dropout DROPOUT]
[--epochs_num EPOCHS_NUM] [--report_steps REPORT_STEPS] [--seed SEED]
[--pooling {mean,max,first,last}] [--tokenizer {bert,char,space}] [--soft_targets]
[--soft_alpha SOFT_ALPHA]

--output_model_path就是模型微调后需要保存的路径，即得到finetuned_model.bin模型
