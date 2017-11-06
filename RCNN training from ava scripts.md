###  从AVA json 开始训练 RCNN
步骤：
1. 下载ava导出的jsonlist
2. 使用json to reccordio tool 转换为recordio
3. 准备对应的class的csv ，一行一个label
4. 运行 `python script/vgg_blued.sh` 开始训练

#### [json to reccordio tool](https://github.com/likelyzhao/RCNN/blob/dev-ava-train/json2rec.py)

工具中调用util 中的 `jsonPack`,`jsonUnpack`函数

输入的参数为：

||||
|---|---|---|
|jsonfile|string(in)|输入的ava输出jsonfile|
|prefix|string(in)|输出的recordio文件名|

输出为recordio的两个文件，分别是{frefix}.idx,{frefix}.rec

