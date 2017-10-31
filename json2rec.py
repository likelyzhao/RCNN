from __future__ import print_function
import argparse
import mxnet as mx
from rcnn.config import config, default, generate_config
from rcnn.utils.pack_json import jsonPack,jsonUnpack


def parse_args():
    parser = argparse.ArgumentParser(description='creat recordio from ava json file')
    # general
    parser.add_argument('--jsonfile', help='json file created by ava',type=str)
    parser.add_argument('--prefix', help='prefix name of output ', default="ava_dect", type=str)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)
    f =open(args.jsonfile)
    import os
    record = mx.recordio.MXIndexedRecordIO(args.prefix + '.idx', args.prefix + '.rec','w')
    idx =0
    for line in f.readlines():
        import json
        dict = json.loads(line)
        import urllib2
        try:
            buf = urllib2.urlopen(dict['url'].strip(),timeout=10).read()
        except :
            print("time out")
            continue
        import numpy as np
        buf = np.fromstring(buf, dtype=np.uint8)
        print(len(buf))
        import cv2
        try:
            img = cv2.imdecode(buf,-1)
            ret, buf = cv2.imencode('.jpg', img)
        except :
            continue
        pack_s = jsonPack(line, buf.tostring())
        record.write_idx(idx, pack_s)
        idx +=1
    print(idx)
    record.close()
    print("total number of training set is " + str(idx) + '\n')
    record = mx.recordio.MXIndexedRecordIO(args.prefix + '.idx', args.prefix + '.rec', 'r')
    item = record.read_idx(9)
    anno, img = jsonUnpack(item)
    import numpy as np
    import json
    print(json.dumps(anno))



if __name__ == '__main__':
    main()
