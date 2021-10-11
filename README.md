## AdaptiveLogSoftmaxWithLoss-Paddle

新增`AdaptiveLogSoftmaxWithLoss`并测试

```
cd AdaptiveLogSoftmaxWithLoss-Paddle
python main.py --epochs 6
```

ref：[adaptivesoftmax_torch](https://github.com/peterzhang2029/adaptivesoftmax_torch)

### For adaptive softmax:
#### Train:
| epoch | epoch time | valid loss | valid ppl |
| - | - | - | - |
| end of epoch   1 | time: 307.37s | valid loss  5.70 | valid ppl  299.94
| end of epoch   2 | time: 308.61s | valid loss  5.42 | valid ppl   225.27
| end of epoch   3 | time: 308.42s | valid loss  5.28 | valid ppl   196.50
| end of epoch   4 | time: 310.00s | valid loss  5.18 | valid ppl   177.47
| end of epoch   5 | time: 309.34s | valid loss  5.12 | valid ppl   167.89
| end of epoch   6 | time: 309.14s | valid loss  5.08 | valid ppl   160.28

#### Test:

test loss  4.99
test ppl  146.40

all Time_cost: 1853.55s

## **关于作者**
<img src="https://ai-studio-static-online.cdn.bcebos.com/cb9a1e29b78b43699f04bde668d4fc534aa68085ba324f3fbcb414f099b5a042" width="100"/>


| 姓名        |  郭权浩                           |
| --------     | -------- | 
| 学校        | 电子科技大学研2020级     | 
| 研究方向     | 计算机视觉             | 
| BLOG主页        | [DeepHao的 blog 主页](https://blog.csdn.net/qq_39567427?spm=1000.2115.3001.5343) |
| GitHub主页        | [DeepHao的 github 主页](https://github.com/GuoQuanhao) |
