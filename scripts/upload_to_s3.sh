#!/bin/bash

current_dir=$(dirname "$0")
mkdir -p "${current_dir}/RQA"
# 拷贝 root 目录下的所有文件和目录到 RQA，但排除 RQA
dir_a=$(dirname "${current_dir}")/..
rsync -av --progress --exclude 'B/RQA' "${dir_a}/" "${current_dir}/RQA"
tar -czvf RQA.tar.gz RQA

# upload to S3
# coscmd delete RQA/RQA.tar.gz
coscmd upload RQA.tar.gz RQA/

# clean everything
rm -rf "${current_dir}/RQA"
# rm -rf "${current_dir}/RQA.tar.gz"
