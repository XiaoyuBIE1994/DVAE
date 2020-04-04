singularity exec \
                --nv \
                --bind /scratch/virgo/xbie/:/mnt/xbie/ \
                /scratch/virgo/xbie/Simgs/test \
                python /mnt/xbie/Code/rvae-speech/train_model.py \
                /mnt/xbie/Code/rvae-speech/config/group_storn_positive/cfg_storn_softplus.ini

