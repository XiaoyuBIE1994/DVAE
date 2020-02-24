oarsub -S /scratch/virgo/xbie/Code/rvae-speech/script/train_ffnn.sh \
        -l/host=1/gpudevice=1,walltime=90:00:00 \
        -p "cluster='perception' " \
        -t besteffort \
        -t idempotent

oarsub -S /scratch/virgo/xbie/Code/rvae-speech/script/train_rnn_Rec.sh \
        -l/host=1/gpudevice=1,walltime=90:00:00 \
        -p "cluster='perception' " \
        -t besteffort \
        -t idempotent

oarsub -S /scratch/virgo/xbie/Code/rvae-speech/script/train_rnn_NoRec.sh \
        -l/host=1/gpudevice=1,walltime=90:00:00 \
        -p "cluster='perception' " \
        -t besteffort \
        -t idempotent

oarsub -S /scratch/virgo/xbie/Code/rvae-speech/script/train_brnn_Rec.sh \
        -l/host=1/gpudevice=1,walltime=90:00:00 \
        -p "cluster='perception' " \
        -t besteffort \
        -t idempotent

oarsub -S /scratch/virgo/xbie/Code/rvae-speech/script/train_brnn_NoRec.sh \
        -l/host=1/gpudevice=1,walltime=90:00:00 \
        -p "cluster='perception' " \
        -t besteffort \
        -t idempotent