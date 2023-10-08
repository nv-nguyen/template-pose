#!/bin/bash

template_pose_docker() {
  xhost +local:docker;
  docker run -it -d \
    --gpus all \
    --name="template_pose_dev" \
    -p 10002:22 \
    -v /etc/localtime:/etc/localtime:ro \
    -v /dev/input:/dev/input \
    -v "$DATASETS:$HOME/datasets" \
    -v "$Template_Pose_GIT:$HOME/template_pose" \
    --shm-size=32G \
    --workdir $HOME/ \
    --env=DISPLAY \
    --env=XDG_RUNTIME_DIR \
    --env=QT_X11_NO_MITSHM=1 \
    --device=/dev/dri:/dev/dri \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /etc/localtime:/etc/localtime:ro \
    template_pose:latest
    template_pose_docker_attach;
}

template_pose_docker_attach() {
  docker exec -it -e "COLUMNS=$COLUMNS" -e "LINES=$LINES" template_pose_dev /bin/bash
}

