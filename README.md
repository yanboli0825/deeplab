# 1 项目结构
本部分主要用于记录项目结构

## 配置文件
conf文件夹主要用于存放配置文件，下分data和model两个主要的subfolders。每个subfolder下以领域来区别不同的配置文件，例如conf/model/cpath/clam下存放的就是CLAM模型的配置文件。

> 配置文件跟代码应该是完全解耦的，这意味着仅依赖配置文件我们就能够随意更改model，dataloader，optimizer，lr_schedular,search range等，无需修改任意一行代码。

## Wandb
### Server



### Probe
训练过程应该尽可能监控更多的metrics。此外，在完整运行训练程序前，最好以fast模式快速检查一下梯度、权重等是否有效，是否能进行正常的学习。
在运行过程中，最好是log住所有val的原始输出结果（例如以json格式），方便进行二次分析。



### Sweep



# 2 Pipeline

## hyperparams search & cross validation

**hyperparams search:**

方案1——flatten cross validation:

拿到数据集后，先固定划分一部分数据出来作为**held-out test set**, 然后对剩下的数据进行K折划分，形成不同的训练集和验证集。固定使用某一折训练集和验证集，在这批数据上进行Sweep，找到一组最优的超参数best-params。找到best-params后，固定这些参数在K折训练集和验证集上跑出K个模型，这K个模型在训练集、验证集和测试集上的mean±std作为指标。最终交付模型时交付一个K个模型的集成，或者交付在验证集和测试集上表现都较优异的一个模型。该方法比较适合做Research时研究算法的泛化能力。



优点：节省算力

缺点：1. 超参数可能会overfitting验证集；2. 数据利用不充分，测试集上的这批数据相当于被舍弃了



方案2——nested cross validation:

拿到数据集后，先对测试集进行K折划分，这是outer fold，再对训练集进行K折划分，这是inner fold。其中，outer fold用于评估泛化能力，inner fold用于评估超参数。在inner fold选出最好的超参数之后，算一个该组超参数在inner fold中的epoch的平均数（当然，也可以跑epoch * 1.2），然后在所有数据中（包括训练集和验证集）重新refit。



优点：overfitting程度可能会小一些，数据利用充分

缺点：算力成本过高

