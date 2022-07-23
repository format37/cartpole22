## gym docker image
* Cuda
* Pytorch
* Tensorflow-gpu
* Gym
* Stable baselines v3
* Example with training bipedal walker DDPG  
  
server.py - is empty python script with infinity loop  

Train script in file: bipedal_worker.py  
Finally saved model to /usr/local/gym/data/BipedalWalker-v3_DDPG-Mlp.zip  
inside container, which u can see in local path data with tensorboard logs
### requirements
* linux
* docker
### installation
```
sh compose
```
### your_container_id
```
ps docker
```
### connect ssh
```
docker exec -it your_container_id bash
```
### Upload a file to container
```
docker cp ./filename.py your_container_id:/usr/local/gym/
```
### Download a file from container
```
docker cp your_container_id:/usr/local/gym/filename.zip ./
```
### logs
```
docker logs -f your_container_id
```
Expected logs
```
INFO:root:2022-07-22 21:56:45.237860 start

INFO:matplotlib.font_manager:Generating new fontManager, this may take some time...

INFO:root:2022-07-22 21:56:50.851151 learn +

INFO:root:2022-07-22 21:58:44.139580 learn -

/usr/local/lib/python3.6/dist-packages/stable_baselines3/common/evaluation.py:69: UserWarning: Evaluation environment is not wrapped with a ``Monitor`` wrapper. This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. Consider wrapping environment first with ``Monitor`` wrapper.

  UserWarning,

INFO:root:2022-07-22 21:58:46.280575 mean_reward=-123.61 +/- 0.1468232030508024
```