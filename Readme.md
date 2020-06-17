# Requirements:

* python-pcl (https://github.com/strawlab/python-pcl)
* point_cloud_utils (https://github.com/search?q=point_cloud_utils)
* open3d (https://github.com/intel-isl/Open3D)
* dlib
* opencv
* scikit-learn

Test OS: Ubuntu16.04
Python version: 3.6(测试3.5/3.7会造成版本冲突)

For Ubuntu:
    #install python-pcl
    conda install -c sirokujira python-pcl

    # 可能会遇到找不到libboost_system.so.1.54.0 (可选)
    cd /anaconda/envs/your-environment/lib
    run:
    $ ln -s libboost_system.so.1.64.0 libboost_system.so.1.54.0
    $ ln -s libboost_filesystem.so.1.64.0 libboost_filesystem.so.1.54.0
    $ ln -s libboost_thread.so.1.64.0 libboost_thread.so.1.54.0
    $ ln -s libboost_iostreams.so.1.64.0 libboost_iostreams.so.1.54.0


    #install open3d
    $ conda install -c open3d-admin open3d=0.9

    # install opencv/dlib/scikit-learn
    $ conda install -c conda-forge opencv
    $ conda install -c menpo dlib
    $ conda install -c anaconda scikit-learn

# Test:
$ python3.6 fusion.py
