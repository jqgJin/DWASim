# 理论选择、核验及论断边界

检索日期：2026-09-06。使用 nature-academic-search 的多源检索/核验流程、
nature-citation 的逐论断支撑分级，以及 nature-ref-verifier 的字段比对规则。
按作者要求检索数学与机器学习原始文献，不限 Nature/CNS。
学术 MCP 未配置；OpenAlex 本地备用脚本连接失败；Crossref 直接查询未成功。
以下通过出版方、作者公开全文和数据库记录交叉核对，未冒称 Crossref 已验证。

## 入选的三项直接基础

| 文献 | 核对来源与位置 | 对本论文的支撑 | 不能据此声称 |
|---|---|---|---|
| Aronszajn N (1950), Theory of reproducing kernels, Trans. AMS 68(3):337–404, DOI 10.1090/S0002-9947-1950-0051437-7 | AMS 原文索引可读的 p.344，第 (3)–(4) 项；Wisconsin 保存的原文；AMS 引文记录 | 强支撑：PSD 与 Hilbert 特征空间的联系。论文另外给出各组件的直接积分/特征构造 | 原文没有研究 DWASim，也没有证明其分类优越性 |
| Cortes C, Mohri M, Rostamizadeh A (2012), Algorithms for Learning Kernels Based on Centered Alignment, JMLR 13:795–828 | JMLR 官网及全文 Section 2.2、3.2.2 Proposition 9、Appendix B Proposition 21；作者页面/DBLP 核对作者与页码 | 强支撑：中心化、非负核组合 QP、QP 扰动分析。这里明确加入 ridge，定义类平衡目标并重新证明所用界 | 不能将该文针对其他预测器/抽样条件的泛化结论直接移植为 HGB 图节点 kNN 的风险保证 |
| Boyd S, Vandenberghe L (2004), Convex Optimization, Cambridge University Press, DOI 10.1017/CBO9780511804441 | Cambridge 书页、两位作者官网及公开全文 Section 9.1.2；2004 年/作者/ISBN 9780521833783 一致 | 强支撑：强凸性；在闭非负正交域上结合强制性得到唯一最优解。本稿给出针对自身 QP 的变分不等式推导 | 唯一最优只针对已定义的代理目标，不能等同于最优 F1 |

原文链接：

- [Aronszajn 原文](https://www.ams.org/tran/1950-068-03/S0002-9947-1950-0051437-7/S0002-9947-1950-0051437-7.pdf)
- [Aronszajn 大学公开副本](https://pages.stat.wisc.edu/~wahba/stat860public/pdf2/aronszajn.pdf)
- [JMLR 文章](https://www.jmlr.org/papers/v13/cortes12a.html)
- [JMLR 全文](https://www.jmlr.org/papers/volume13/cortes12a/cortes12a.pdf)
- [凸优化作者官网](https://web.stanford.edu/~boyd/cvxbook/)
- [凸优化公开全文](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)

AMS 原文直接打开返回 403，但其检索索引返回了 p.344 原文段落；没有绕过访问限制。
Aronszajn 的 DOI 与标题经 AMS 出版方记录核对，JMLR 未强行编造 DOI。

## 为什么选这些理论

1. **核表示，不强套原始度量。** 相对幅值相似度是归一化直方图交的积分核。
   支持相似度由二值版本经非负幂级数得到；方向相似度是单位向量内积。
   三个组件因此能在同一直接和 Hilbert 空间中解释。
2. **整数幂校准，不加新表示网络。** 每条路径的核取整数次幂对应张量特征，
   所有 PSD 性质保留。只在融合前校准，融合后取单调幂不会改善排序。
3. **排名间隔，不凭空声称泛化。** 固定参数下给出明确扰动界与 top-k 条件。
   小间隔、并列、参数重选都不能从该界得到无条件保证。
4. **强凸目标，不把代理目标包装为 F1 最优。** 对齐实验是保留的科学对照；
   即使最优解唯一，也可能在分类上弱于原方法。

## 不选或不主张的方向

- 不将 Bray–Curtis 当作满足三角不等式的距离；正文给出了反例。
- 不把 PSD 误写成严格正定；重复节点/重复组件可产生零特征值。
- 不引入 PAC-Bayes / VC 或独立同分布的一般风险界装饰固定图实验。
- 不用测试结果挑选特定数据集的“胜出版本”作为未预设主方法。
- 不宣称本文发明 RKHS、核幂闭包或 centered alignment；新内容是适配、
  直接证明、有限校准实现及透明对照。

## 文稿论证结构

类型关系 → 交互轮廓 → 三组件核 → 非负路径融合 → 可选幂校准 → 原有近邻预测。
主文保留核构造、常量组件、幂校准定理；SI 说明边界条件、QP 与扰动证明，
避免为增加定理数量而保留纯定义式“表示充分性定理”。原表示信息边界保留为段落。
