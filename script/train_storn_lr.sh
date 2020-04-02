singularity exec \
                --nv \
                --bind /scratch/virgo/xbie/:/mnt/xbie/ \
                /scratch/virgo/xbie/Simgs/test \
                python /mnt/xbie/Code/rvae-speech/train_model.py \
                /mnt/xbie/Code/rvae-speech/config/group_storn_lr/cfg_storn_lr1e-1.ini

singularity exec \
                --nv \
                --bind /scratch/virgo/xbie/:/mnt/xbie/ \
                /scratch/virgo/xbie/Simgs/test \
                python /mnt/xbie/Code/rvae-speech/train_model.py \
                /mnt/xbie/Code/rvae-speech/config/group_storn_lr/cfg_storn_lr1e-2.ini

singularity exec \
                --nv \
                --bind /scratch/virgo/xbie/:/mnt/xbie/ \
                /scratch/virgo/xbie/Simgs/test \
                python /mnt/xbie/Code/rvae-speech/train_model.py \
                /mnt/xbie/Code/rvae-speech/config/group_storn_lr/cfg_storn_lr1e-3.ini

singularity exec \
                --nv \
                --bind /scratch/virgo/xbie/:/mnt/xbie/ \
                /scratch/virgo/xbie/Simgs/test \
                python /mnt/xbie/Code/rvae-speech/train_model.py \
                /mnt/xbie/Code/rvae-speech/config/group_storn_lr/cfg_storn_lr3e-1.ini

singularity exec \
                --nv \
                --bind /scratch/virgo/xbie/:/mnt/xbie/ \
                /scratch/virgo/xbie/Simgs/test \
                python /mnt/xbie/Code/rvae-speech/train_model.py \
                /mnt/xbie/Code/rvae-speech/config/group_storn_lr/cfg_storn_lr3e-2.ini

singularity exec \
                --nv \
                --bind /scratch/virgo/xbie/:/mnt/xbie/ \
                /scratch/virgo/xbie/Simgs/test \
                python /mnt/xbie/Code/rvae-speech/train_model.py \
                /mnt/xbie/Code/rvae-speech/config/group_storn_lr/cfg_storn_lr3e-3.ini



