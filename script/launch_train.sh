oarsub -S /scratch/virgo/xbie/Code/rvae-speech/script/train_vae.sh \
        -l/host=1/gpudevice=1,walltime=90:00:00 \
        -p "cluster='perception' OR cluster='kinovis'" \
        -t besteffort \
        -t idempotent

oarsub -S /scratch/virgo/xbie/Code/rvae-speech/script/train_storn.sh \
        -l/host=1/gpudevice=1,walltime=90:00:00 \
        -p "cluster='perception' OR cluster='kinovis'" \
        -t besteffort \
        -t idempotent

oarsub -S /scratch/virgo/xbie/Code/rvae-speech/script/train_srnn.sh \
        -l/host=1/gpudevice=1,walltime=90:00:00 \
        -p "cluster='perception' OR cluster='kinovis'" \
        -t besteffort \
        -t idempotent

oarsub -S /scratch/virgo/xbie/Code/rvae-speech/script/train_dks.sh \
        -l/host=1/gpudevice=1,walltime=90:00:00 \
        -p "cluster='perception' OR cluster='kinovis'" \
        -t besteffort \
        -t idempotent

oarsub -S /scratch/virgo/xbie/Code/rvae-speech/script/train_vrnn.sh \
        -l/host=1/gpudevice=1,walltime=90:00:00 \
        -p "cluster='perception' OR cluster='kinovis'" \
        -t besteffort \
        -t idempotent

oarsub -S /scratch/virgo/xbie/Code/rvae-speech/script/train_rvae_BiNoRecZ.sh \
        -l/host=1/gpudevice=1,walltime=90:00:00 \
        -p "cluster='perception' OR cluster='kinovis'" \
        -t besteffort \
        -t idempotent

oarsub -S /scratch/virgo/xbie/Code/rvae-speech/script/train_rvae_BiRecZ.sh \
        -l/host=1/gpudevice=1,walltime=90:00:00 \
        -p "cluster='perception' OR cluster='kinovis'" \
        -t besteffort \
        -t idempotent

oarsub -S /scratch/virgo/xbie/Code/rvae-speech/script/train_rvae_NoRecZ.sh \
        -l/host=1/gpudevice=1,walltime=90:00:00 \
        -p "cluster='perception' OR cluster='kinovis'" \
        -t besteffort \
        -t idempotent

oarsub -S /scratch/virgo/xbie/Code/rvae-speech/script/train_rvae_RecZ.sh \
        -l/host=1/gpudevice=1,walltime=90:00:00 \
        -p "cluster='perception' OR cluster='kinovis'" \
        -t besteffort \
        -t idempotent