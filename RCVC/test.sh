# python test.py --cuda true --cuda_device 2 --worker 1  --test_config dataset_config_example.json --output_json_result_path  epoch3_DCVC_result_psnr.json 
# --recon_bin_path recon_bin_folder_psnr --model_type psnr  --i_frame_model_name cheng2020-anchor  --i_frame_model_path  checkpoints/cheng2020-anchor-3-e49be189.pth.tar  checkpoints/cheng2020-anchor-4-98b0b468.pth.tar   checkpoints/cheng2020-anchor-5-23852949.pth.tar   checkpoints/cheng2020-anchor-6-4c052b1a.pth.tar   --model_path /mnt/DVC_DCVC/DCVC/Result/dcvc_b/snapshot/epoch3.model  checkpoints/model_dcvc_quality_1_psnr.pth checkpoints/model_dcvc_quality_2_psnr.pth checkpoints/model_dcvc_quality_3_psnr.pth  --write_stream True

# python test.py --i_frame_model_name cheng2020-anchor  --i_frame_model_path  checkpoints/cheng2020-anchor-3-e49be189.pth.tar  checkpoints/cheng2020-anchor-4-98b0b468.pth.tar   checkpoints/cheng2020-anchor-5-23852949.pth.tar   checkpoints/cheng2020-anchor-6-4c052b1a.pth.tar  --test_config     dataset_config_example.json  --cuda true --cuda_device 2   --worker 1   --output_json_result_path  PreModel_DCVC_result_psnr.json    --model_type psnr  --recon_bin_path recon_bin_folder_psnr --model_path checkpoints/model_dcvc_quality_0_psnr.pth  checkpoints/model_dcvc_quality_1_psnr.pth checkpoints/model_dcvc_quality_2_psnr.pth checkpoints/model_dcvc_quality_3_psnr.pth  



#用最短的时间先测试一下
python test.py --i_frame_model_name cheng2020-anchor --i_frame_model_path  checkpoints/cheng2020-anchor-3-e49be189.pth.tar --test_config  dataset_config_example.json  --cuda true --cuda_device 3   --worker 1  --output_json_result_path  ./a_UVG_results/RCVC_l_epoch11_result_psnr.json    --model_type psnr  --recon_bin_path recon_bin_folder_psnr --model_path   /mnt/DVC_DCVC/RCVC1/a_Train_Result/RCVC_l/snapshot/epoch11.model

# /mnt/RCVC/RCVC1/Result/RCVC_work59/epoch15.model  /mnt/RCVC/RCVC1/Result/RCVC_c/snapshot/epoch8.model /mnt/RCVC/RCVC1/Result/RCVC_b/snapshot/epoch9.model