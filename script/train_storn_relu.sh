# singularity exec \
#                 --nv \
#                 --bind /scratch/virgo/xbie/:/mnt/xbie/ \
#                 /scratch/virgo/xbie/Simgs/test \
#                 python /mnt/xbie/Code/rvae-speech/train_model.py /mnt/xbie/Code/rvae-speech/config/cfg_storn_act=relu_dense=1.ini

singularity exec \
                --nv \
                --bind /scratch/virgo/xbie/:/mnt/xbie/ \
                /scratch/virgo/xbie/Simgs/test \
                python /mnt/xbie/Code/rvae-speech/train_model.py /mnt/xbie/Code/rvae-speech/config/cfg_storn_act=relu_dense=2.ini

singularity exec \
                --nv \
                --bind /scratch/virgo/xbie/:/mnt/xbie/ \
                /scratch/virgo/xbie/Simgs/test \
                python /mnt/xbie/Code/rvae-speech/train_model.py /mnt/xbie/Code/rvae-speech/config/cfg_storn_act=relu_dense=3.ini

singularity exec \
                --nv \
                --bind /scratch/virgo/xbie/:/mnt/xbie/ \
                /scratch/virgo/xbie/Simgs/test \
                python /mnt/xbie/Code/rvae-speech/train_model.py /mnt/xbie/Code/rvae-speech/config/cfg_storn_act=relu_dense=4.ini