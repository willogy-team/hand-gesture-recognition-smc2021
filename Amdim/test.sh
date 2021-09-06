
NC=7
NE=30
BS=32
LR=0.0001
NDF=256
N_RKHS=2048
N_DEPTH=8
TCLIP=20.0
PT="pretrainImageNet"
TRAIN_SCENE="Scene1"
TEST_SCENE="Scene2"

python test.py --n_classes ${NC} --batch_size ${BS} --dataset QA10SCENES --input_dir "/media/data-huy/dataset/QADataset/SceneCategory_Frame_final_7classes" --checkpoint_path "./runs/mfh_classifier_${PT}_${NC}_${NE}_${BS}_${LR}_${NDF}_${N_RKHS}_${N_DEPTH}_${TCLIP}.pth" --train_scene ${TRAIN_SCENE} --test_scene ${TEST_SCENE} --suffix_name "test_mfh_classifier_${PT}_${NC}_${NE}_${BS}_${LR}_${NDF}_${N_RKHS}_${N_DEPTH}_${TCLIP}" --amp > "./output/mfh_classifier_${PT}_${NC}_${NE}_${BS}_${LR}_${NDF}_${N_RKHS}_${N_DEPTH}_${TCLIP}_test.txt"