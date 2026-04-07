python main.py --dataset CIFAR10 --proxy_model ResNet --victim_model SimpleNN\
                --total_dataset_size 50000 --poison_dataset_size 1 --proxy_epochs 3 --victim_epochs 3\
                --attack_name pickme

python main.py --dataset CIFAR10 --proxy_model ResNet --victim_model SimpleNN\
                --total_dataset_size 50000 --poison_dataset_size 10 --proxy_epochs 3 --victim_epochs 3\
                --attack_name pickme

python main.py --dataset CIFAR10 --proxy_model ResNet --victim_model SimpleNN\
                --total_dataset_size 50000 --poison_dataset_size 100 --proxy_epochs 3 --victim_epochs 3\
                --attack_name pickme 

python main.py --dataset CIFAR10 --proxy_model ResNet --victim_model SimpleNN\
                --total_dataset_size 50000 --poison_dataset_size 1000 --proxy_epochs 3 --victim_epochs 3\
                --attack_name pickme

python main.py --dataset CIFAR10 --proxy_model ResNet --victim_model SimpleNN\
                --total_dataset_size 50000 --poison_dataset_size 5000 --proxy_epochs 3 --victim_epochs 3\
                --attack_name pickme