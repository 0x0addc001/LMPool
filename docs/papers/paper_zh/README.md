# LMPool Paper 中文版

`paper_zh.tex` 是 `../paper/example_paper.tex` 的中文翻译版。它保留相同的公式、实验数据、参考文献键和图表引用；图内英文标签继续复用英文原图，中文图注解释其含义。`figures/` 和 `references.bib` 由`sync_assets.sh` 从英文论文同步。中文版使用`unsrtnat`，参考文献按首次引用顺序编号。它支持两种编译引擎：`pdflatex` 使用 CJKutf8 回退字体，XeLaTeX 使用 Fandol 字体。英文论文的图、数据或 bibliography 更新后，在仓库根目录运行：

```bash
bash docs/papers/paper_zh/sync_assets.sh
```

默认 `pdflatex` 构建方式：

```bash
cd docs/papers/paper_zh
pdflatex paper_zh.tex
bibtex paper_zh
pdflatex paper_zh.tex
pdflatex paper_zh.tex
```

也可以使用 XeLaTeX：

```bash
cd docs/papers/paper_zh
xelatex paper_zh.tex
bibtex paper_zh
xelatex paper_zh.tex
xelatex paper_zh.tex
```

目录中的 `latexmkrc` 默认选择 `pdflatex`。若使用 XeLaTeX，可运行：

```bash
latexmk -xelatex paper_zh.tex
```
