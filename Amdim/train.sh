CUDA_VISIBLE_DEVICES=0

NC=7
NE=30
BS=32
LR=0.0001
NDF=256
N_RKHS=2048
N_DEPTH=8
TCLIP=20.0
PT="pretrainImageNet"
SCENE="Scene1"

# Use "--classifiers" if you want to only train the classifier part of the network; otherwise, remove this argument to only train the self-supervised part of the network.

python train.py --classifiers --n_classes ${NC} --n_epochs ${NE} --batch_size ${BS} --learning_rate ${LR} --ndf ${NDF} --n_rkhs ${N_RKHS} --n_depth ${N_DEPTH} --tclip ${TCLIP} --dataset QA10SCENES --input_dir "/media/data-huy/dataset/QADataset/SceneCategory_Frame_final_7classes" --cpt_load_path "./models/amdim_ndf256_rkhs2048_rd10.pth" --cpt_name "mfh_classifier_${PT}_${NC}_${NE}_${BS}_${LR}_${NDF}_${N_RKHS}_${N_DEPTH}_${TCLIP}.pth" --run_name "mfh_classifier_${PT}_${NC}_${NE}_${BS}_${LR}_${NDF}_${N_RKHS}_${N_DEPTH}_${TCLIP}" --train_scene ${SCENE} --test_scene ${SCENE} --amp