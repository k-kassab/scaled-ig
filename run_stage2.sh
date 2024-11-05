if [ "$1" = "" ]; then
    config_name=stage2_default.yaml
else
    config_name=$1
fi

echo running stage2.py with config $config_name, pretraining...
accelerate launch stage2.py --pretrain --config $config_name
echo running stage2.py with config $config_name, training...
accelerate launch stage2.py --train --config $config_name
