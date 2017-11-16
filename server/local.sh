killall python

logdir="./log"
if [ ! -d "$logdir" ]; then
  mkdir "$logdir"
fi
if [ -d "test" ]; then
  rm -rf "test"
fi

nohup python train_ps.py --task_index=0 --job_name=worker --cluster_conf="config/cluster_conf.json" --log_dir="test" >$logdir/log.log0 2>&1 &
nohup python train_ps.py --task_index=1 --job_name=worker --cluster_conf="config/cluster_conf.json" --log_dir="test" >$logdir/log.log1 2>&1 &
nohup python train_ps.py --task_index=2 --job_name=worker --cluster_conf="config/cluster_conf.json" --log_dir="test" >$logdir/log.log2 2>&1 &
nohup python train_ps.py --task_index=0 --job_name=ps --cluster_conf="config/cluster_conf.json" >$logdir/log.log3 2>&1 &
