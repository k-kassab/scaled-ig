if [ "$1" = "" ]; then
    config_name=stage1_default.yaml
else
    config_name=$1
fi

echo running stage1.py with config $config_name, pretraining...
accelerate launch stage1.py --pretrain --config $config_name
echo running stage1.py with config $config_name, training...
accelerate launch stage1.py --train --config $config_name
