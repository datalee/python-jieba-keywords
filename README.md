# python-jieba-keywords

一个更易维护、可复用的 `jieba` 关键词提取示例（Python 3）。

## 优化点

- 把 TF-IDF 和 TextRank 的逻辑封装到 `KeywordExtractor`，避免示例代码重复。
- 复用 `jieba.analyse.TFIDF()` 与 `jieba.analyse.TextRank()` 实例，适合批量处理场景。
- 用 `dataclass` 统一结果结构，方便后续扩展（如排序、序列化、写入数据库）。
- 使用 Python 3 输出格式，替代旧版 `print item[0], item[1]`。

## 安装

```bash
pip install jieba
```

## 运行示例

```bash
python keyword_extractor.py
```

## 原始文本

```text
该同学来电反映学校食堂趁刮台风涨价，建议拨打12345
```

## 主要代码

```python
from keyword_extractor import KeywordExtractor

content = "该同学来电反映学校食堂趁刮台风涨价，建议拨打12345"
extractor = KeywordExtractor(top_k=10, allow_pos=("n", "nr", "ns"))

tfidf_keywords = extractor.extract_tfidf(content)
textrank_keywords = extractor.extract_textrank(content)
```
