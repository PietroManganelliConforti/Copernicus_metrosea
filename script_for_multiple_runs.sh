


python3 main.py --batch_size 64 --batch_size_test 256 --alpha 0.0 --lr 1e-3 --num_epochs 1 --augmentations 0 --test_name baseline

python3 main.py --batch_size 64 --batch_size_test 256 --alpha 0.0 --lr 1e-4 --num_epochs 1 --augmentations 0 --test_name lr_1e-4

python3 main.py --batch_size 64 --batch_size_test 256 --alpha 0.2 --lr 1e-3 --num_epochs 1 --augmentations 0 --test_name test_loss_alpha_0.2

python3 main.py --batch_size 64 --batch_size_test 256 --alpha 0.0 --lr 1e-3 --num_epochs 1 --augmentations 1 --test_name test_augmentations1

python3 main.py --batch_size 64 --batch_size_test 256 --alpha 0.0 --lr 1e-3 --num_epochs 1 --augmentations 2 --test_name test_augmentations2

python3 main.py --batch_size 64 --batch_size_test 256 --alpha 0.0 --lr 1e-3 --num_epochs 1 --augmentations 3 --test_name test_augmentations3

python3 main.py --batch_size 64 --batch_size_test 256 --alpha 0.0 --lr 1e-3 --num_epochs 1 --augmentations 4 --test_name test_augmentations4


# #python3 main.py --batch_size 64 --batch_size_test 256 --alpha 0.0 --lr 1e-3 --num_epochs 50 --small_net_flag True --augmentations 0 --test_name small_net