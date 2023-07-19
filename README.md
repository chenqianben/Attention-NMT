# NMT Assignment
Note: Heavily inspired by the https://github.com/pcyin/pytorch_nmt repository

#### Attention mechanism

    

|  Attention mechanism       |                              Formula                                |
|----------------------------|---------------------------------------------------------------------|
|Dot product attention       |$$e_{t,i} = s_t^T h_i$$                                              |
|Multiplicative attention    |$$e_{t,i} = s_t^T W h_i$$                                            |
|Additive attention          |$$e_{t,i} = = v^T (W_1 h_i + W_2 s_t)$$                              |
|Location-Aware Attention    |$$e_{t,i} = = W_1 h_i + W_2 s_t + W_3 f_i$$                          |
|Scaled Dot-Product Attention|$$e_{t,i} = = \frac{s_t^T h_i}{\sqrt{d}} $$                          |
|Multi-head Attention        |$$e_{t,i} = = Concat(s_{t,1}^T h_{i,1},...,s_{t,N}^T h_{i,N}) $$     |



#### TAKE AWAY

- **LOVE** Python 3.5 type hints, reminds me of scala :D
- Always check **bias** in Linear layer, default is True

```python
# (2, 4, 1, 4, 2).squeeze() -> (2, 4, 4, 2). specific dimension will squeeze this dim
# (2, 4, 4, 2).unsqueeze(-1) -> (2, 4, 4, 2, 1)
# torch.cat(( T(2, 4), T(2, 3) ), dim=1) -> T(2, 7)
# torch.split(T(5, 2, 1), 1, dim=0) # split dimension 0, 1 each : [T(2, 1)] * 5
# torch.bmm( T(a, b, c), T(a, c, d)) -> T(a, b, d)
```
