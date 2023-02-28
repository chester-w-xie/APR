# APR
code for the paper: . "Few-shot class-incremental audio classification using adaptively-refined prototypes", Submitted to INTERSPEECH 2023'


First, prepare the dataset according to the instructions here https://github.com/chester-w-xie/FCAC_datasets  

Then run the following command：
```
python main_SPPR_FSC89.py --seed 1688 --way 5 --shot 5 --session 7 --trials 100 --lr 0.1 --lr-scheduler cos \
--early_stop_tol 100 --batch_size 128 --pretrained --gpu-ids 2  --base_epochs 100 \
--metapath path to the FSC-89-meta folder \
--datapath path to the FSD-MIX-CLIPS_data folder --data_type audio --setup mini \
--batch_task 3
```

```
python main_SPPR_Nsynth.py --seed 1688 --way 5 --shot 5 --session 10 --trials 100 --lr 0.1 --lr-scheduler cos \
--early_stop_tol 50 --batch_size 128 --pretrained --gpu-ids 2  --base_epochs 100 \
--metapath path to the The_NSynth_Dataset folder \
--num_class 100 --base_class 55 --batch_task 3
```
