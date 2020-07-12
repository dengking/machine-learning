# parser的实现分析

- `tree.py`
- `probability.py`
- `grammar.py`
- parse

## 功能分析

根据用户指定的grammar，构造parser。



## 支持的语法

- CFG
- PCFG(**probabilistic context free grammar** )





首先需要考虑的如何来表示[CFG](https://en.wikipedia.org/wiki/Context-free_grammar)? 由`grammar.py`实现。

## Nonterminal

|                               |                                                              |
| ----------------------------- | ------------------------------------------------------------ |
| `class Nonterminal`           | A non-terminal symbol for a context free grammar, immutable and hashable |
| `class FeatStructNonterminal` |                                                              |



## Productions

|                                          |                                 |
| ---------------------------------------- | ------------------------------- |
| `class Production`                       |                                 |
| `class DependencyProduction(Production)` | A dependency grammar production |
| `class ProbabilisticProduction`          |                                 |



## Grammars

|                                        |                                                              |
| -------------------------------------- | ------------------------------------------------------------ |
| `class CFG`                            | A context-free grammar.  A grammar consists of a start state and a set of productions. |
| `class FeatureGrammar(CFG)`            | A dependency grammar production                              |
| `class DependencyGrammar`              |                                                              |
| `class ProbabilisticDependencyGrammar` |                                                              |



## grammar

### `class RecursiveDescentParser(ParserI)`





### `class TransitionParser(ParserI)`



### `class BottomUpProbabilisticChartParser(ParserI)`





### `class MaltParser(ParserI)`





### `class GenericStanfordParser(ParserI)`



### `class ViterbiParser(ParserI)`





### `class ChartParser(ParserI)`



### `class GenericCoreNLPParser(ParserI, TokenizerI, TaggerI)`



### `class BllipParser(ParserI)`





### `class ShiftReduceParser(ParserI)`

