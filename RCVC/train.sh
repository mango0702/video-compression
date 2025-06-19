#ROOT=savecode/
#export PYTHONPATH=$PYTHONPATH:$ROOT
# mkdir Result/Train/snapshot
# #CUDA_VISIBLE_DEVICES=0,1 python -u $ROOT/main.py --log log.txt --config config.json
# CUDA_VISIBLE_DEVICES=2 python -u main_2_Copy1.py --log Result/Train/DCVC_log256_mytrain_hispretrain.txt --config config_Copy1.json

mkdir a_Train_Result/RCVC_u
mkdir a_Train_Result/RCVC_u/snapshot
CUDA_VISIBLE_DEVICES=0 python train.py --batch_size 16 --i_frame_model_name cheng2020-anchor  --i_frame_model_path checkpoints/cheng2020-anchor-3-e49be189.pth.tar --dataset /mnt/DVC/data/vimeo_septuplet --num_workers 1 --pretrain_DCVC  /mnt/DVC_DCVC/DCVC/checkpoints/model_dcvc_quality_0_psnr.pth --pretrain_DVC /mnt/DVC_DCVC//RCVC1/a_delete/Result/256.model
# --pretrain /mnt/DVC_DCVC/RCVC1/a_delete/Result/RCVC_b/snapshot/epoch98.model 
# --pretrain_DCVC  /mnt/DVC_DCVC/DCVC/checkpoints/model_dcvc_quality_0_psnr.pth

# --pretrain /mnt/DVC_DCVC/DCVC/checkpoints/model_dcvc_quality_0_psnr.pth
# --pretrain /mnt/RCVC/RCVC1/Result/256.model
# 
#--pretrain /mnt/DVC_DCVC/DCVC/checkpoints/model_dcvc_quality_0_psnr.pth
# --pretrain_DVC /mnt/DVC_DCVC//RCVC1/a_delete/Result/256.model
# --pretrain /mnt/RCVC/RCVC1/Result/RCVC_f/snapshot/epoch6.model