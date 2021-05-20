#!/bin/bash
#SBATCH -t 1-12
#SBATCH -c 4
#SBATCH --job-name=tin
#SBATCH --mem=10000
#SBATCH --account=rrg-bashivan
#SBATCH --gres=gpu:2
# #SBATCH --array=0-53


module load python/3.6
source /home/bashivan/envs/torch/bin/activate

# download and unzip dataset
# wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
rsync --progress -av "/home/bashivan/scratch/results/afd/data/tiny-imagenet-200" $SLURM_TMPDIR
cd "$SLURM_TMPDIR"
unzip "$SLURM_TMPDIR/tiny-imagenet-200/tiny-imagenet-200.zip" 

current="$(pwd)/tiny-imagenet-200"

# training data
cd $current/train
for DIR in $(ls); do
   cd $DIR
   rm *.txt
   mv images/* .
   rm -r images
   cd ..
done

# validation data
cd $current/val
annotate_file="val_annotations.txt"
length=$(cat $annotate_file | wc -l)
for i in $(seq 1 $length); do
    # fetch i th line
    line=$(sed -n ${i}p $annotate_file)
    # get file name and directory name
    file=$(echo $line | cut -f1 -d" " )
    directory=$(echo $line | cut -f2 -d" ")
    mkdir -p $directory
    mv images/$file $directory
done
rm -r images
echo "done"

python /home/bashivan/codes/downloads/TRADES_fork/train_trades_tiny_imagenet.py \
--save_path='/scratch/bashivan/results/afd' \
--data_path="$SLURM_TMPDIR/tiny-imagenet-200"
