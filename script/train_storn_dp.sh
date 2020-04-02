singularity exec \
                --nv \
                --bind /scratch/virgo/xbie/:/mnt/xbie/ \
                /scratch/virgo/xbie/Simgs/test \
                python /mnt/xbie/Code/rvae-speech/train_model.py \
                /mnt/xbie/Code/rvae-speech/config/group_storn_dropout/cfg_storn_dp2.ini

singularity exec \
                --nv \
                --bind /scratch/virgo/xbie/:/mnt/xbie/ \
                /scratch/virgo/xbie/Simgs/test \
                python /mnt/xbie/Code/rvae-speech/train_model.py \
                /mnt/xbie/Code/rvae-speech/config/group_storn_dropout/cfg_storn_dp4.ini

singularity exec \
                --nv \
                --bind /scratch/virgo/xbie/:/mnt/xbie/ \
                /scratch/virgo/xbie/Simgs/test \
                python /mnt/xbie/Code/rvae-speech/train_model.py \
                /mnt/xbie/Code/rvae-speech/config/group_storn_dropout/cfg_storn_dp6.ini

singularity exec \
                --nv \
                --bind /scratch/virgo/xbie/:/mnt/xbie/ \
                /scratch/virgo/xbie/Simgs/test \
                python /mnt/xbie/Code/rvae-speech/train_model.py \
                /mnt/xbie/Code/rvae-speech/config/group_storn_dropout/cfg_storn_dp8.ini

