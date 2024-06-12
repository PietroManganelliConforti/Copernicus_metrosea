#loss_alpha
#nohup sh script_for_multiple_runs.sh  > /dev/null 2>&1 &

#ps aux | grep script and then kill, if needed


run_mains()
{
python3 main.py --batch_size 32 --batch_size_test 256 --loss_alpha 0.0 --ema_alpha 0.0 --lr 1e-4 --num_epochs 50 --augmentations 0 --test_name test_LSTM --dim 1D 

python3 main.py --batch_size 32 --batch_size_test 256 --loss_alpha 0.0 --ema_alpha 0.9 --lr 1e-4 --num_epochs 50 --augmentations 4 --test_name test_augmentationsConEma

python3 main.py --batch_size 32 --batch_size_test 256 --loss_alpha 0.0 --ema_alpha 0.9 --lr 1e-4 --num_epochs 50 --augmentations 0 --test_name solo_ema
    
python3 main.py --batch_size 32 --batch_size_test 256 --loss_alpha 0.0 --ema_alpha 0.0 --lr 1e-4 --num_epochs 50 --augmentations 0 --test_name baseline_oggi

python3 main.py --batch_size 32 --batch_size_test 256 --loss_alpha 0.0 --ema_alpha 0.0 --lr 1e-4 --num_epochs 50 --augmentations 4 --test_name test_augmentations4_2try

python3 main.py --batch_size 32 --batch_size_test 256 --loss_alpha 0.0 --ema_alpha 0.0 --lr 1e-4 --num_epochs 50 --augmentations 5 --test_name test_augmentations5_2try


# python3 main.py --batch_size 32 --batch_size_test 256 --loss_alpha 0.02 --ema_alpha 0.0 --lr 1e-4 --num_epochs 50 --augmentations 0 --test_name test_loss_alpha_0.02_2try

#python3 main.py --batch_size 32 --batch_size_test 256 --loss_alpha 0.0 --ema_alpha 0.0 --lr 1e-3 --num_epochs 50 --augmentations 0 --test_name lr_1e-3
#
#python3 main.py --batch_size 32 --batch_size_test 256 --loss_alpha 0.0 --ema_alpha 0.0 --lr 1e-4 --num_epochs 50 --augmentations 1 --test_name test_augmentations1
#
#python3 main.py --batch_size 32 --batch_size_test 256 --loss_alpha 0.0 --ema_alpha 0.0 --lr 1e-4 --num_epochs 50 --augmentations 2 --test_name test_augmentations2
#
#python3 main.py --batch_size 32 --batch_size_test 256 --loss_alpha 0.0 --ema_alpha 0.0 --lr 1e-4 --num_epochs 50 --augmentations 3 --test_name test_augmentations3
#
#python3 main.py --batch_size 32 --batch_size_test 256 --loss_alpha 0.0 --ema_alpha 0.0 --lr 1e-4 --num_epochs 50 --augmentations 4 --test_name test_augmentations4
#
#python3 main.py --batch_size 32 --batch_size_test 256 --loss_alpha 0.0 --ema_alpha 0.0 --lr 1e-4 --num_epochs 50 --augmentations 5 --test_name test_augmentations5
#
#python3 main.py --batch_size 32 --batch_size_test 256 --loss_alpha 0.0 --ema_alpha 0.0 --lr 1e-4 --num_epochs 50 --augmentations 6 --test_name test_augmentations6
#
#python3 main.py --batch_size 32 --batch_size_test 256 --loss_alpha 0.0 --ema_alpha 0.0 --lr 1e-4 --num_epochs 50 --augmentations 7 --test_name test_augmentations7

}



nohup run_mains > /dev/null 2>&1 &





