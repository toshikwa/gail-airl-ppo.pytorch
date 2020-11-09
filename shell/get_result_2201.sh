#! /bin/bash
scp -r -P 2201 -i ~/Server/id_rsa_2201 zhangsongyuan@101.6.32.246:~/Downloads/Result.tar.gz ../Result.tar.gz && tar -zxvf ../Result.tar.gz && rm ../Result.tar.gz
