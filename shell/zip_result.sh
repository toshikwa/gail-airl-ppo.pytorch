#! /bin/bash
tar -zcvf Result.tar.gz ../logs/ && echo "zip finished"
mv Result.tar.gz ~/Downloads/Result.tar.gz && echo "tar finished"
