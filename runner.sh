torchrun --standalone --nproc_per_node=2 hicinterpolate.py --distributed --train --test --config config_64_set_6_local
torchrun --standalone --nproc_per_node=2 hicinterpolate.py --distributed --train --test --config config_128_set_7_local

torchrun --standalone --nproc_per_node=2 hicinterpolate.py --distributed --train --test --config config_64_set_17_local