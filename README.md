# AL_demo

This is a short demonstration of different sequential sampling strategies. It includes
1. Random sampling
2. Uncertainty sampling
3. Variance reduction (IMSE: Integrated Mean Squared Error)
4. Maximin distance
5. Querry by Committee
6. Partitioned Active Learning

Execute "aldemo.py" in your terminal, then you will be asked to choose a strategy from above
Once you choose a strategy, you will see how each strategy behaves with a simple 2D function (from [Gramacy et al. (2009)](https://www.tandfonline.com/doi/abs/10.1198/TECH.2009.0015?casa_token=eQ5QUYXFHq0AAAAA:glynKn8IxE5d-GksdYadnCwzcrN30oMF--s1gRg8k-BZ6_2ouTr6x224L1nksyvAIzd-jBmsarjzsA)) approximation.
A new sample will be added to the training set and the model will be updated for every hitting enter.

For more about "Partitioned Active Learning", please refer to [our paper](https://arxiv.org/abs/2105.08547).
```
@article{lee2021partitioned,
  title={Partitioned Active Learning for Heterogeneous Systems},
  author={Lee, Cheolhei and Wang, Kaiwen and Wu, Jianguo and Cai, Wenjun and Yue, Xiaowei},
  journal={arXiv preprint arXiv:2105.08547},
  year={2021}
}
```

The code additionally requires the following packages (and checked with):
```
python > 3.9
numpy > 1.21
scipy > 1.7
scikit-learn > 1.0
```
