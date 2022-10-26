#### Creating viewpoints to generate templates

Download blender
```
BLENDER_LIB_PATH=/home/nguyen/Documents/code/blender_lib
mkdir $BLENDER_LIB_PATH
curl https://download.blender.org/release/Blender2.82/blender-2.82-linux64.tar.xz -o $BLENDER_LIB_PATH/blender-2.82-linux64.tar.xz
tar xf $BLENDER_LIB_PATH/blender-2.82-linux64.tar.xz -C $BLENDER_LIB_PATH
$BLENDER_LIB_PATH/blender-2.82-linux64/blender -b --python ./lib/poses/predefined_poses/create_viewpoints.py
```

Create viewpoints by subdividing icosphere:
```
$BLENDER_LIB_PATH/blender-2.82-linux64/2.82/python/bin/python3.7m ./lib/poses/predefined_poses/create_viewpoints.py
```