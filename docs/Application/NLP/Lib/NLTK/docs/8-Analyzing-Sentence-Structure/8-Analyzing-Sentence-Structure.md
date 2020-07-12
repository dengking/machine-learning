[TOC]



# 8. Analyzing Sentence Structure

Earlier chapters focused on words: how to identify them, analyze their structure, assign them to lexical categories, and access their meanings. We have also seen how to identify **patterns** in **word sequences** or n-grams. However, these methods only scratch the surface of the complex constraints that govern sentences. We need a way to deal with the ambiguity that **natural language** is famous for. We also need to be able to cope with the fact that there are an unlimited number of possible sentences, and we can only write finite programs to analyze their structures and discover their meanings.

> NOTE: As far as grammar is concerned, formal language is much simpler than natural language

The goal of this chapter is to answer the following questions:

1. How can we use a **formal grammar** to describe the structure of an unlimited set of sentences?
2. How do we represent the structure of sentences using **syntax trees**?
3. How do **parsers** analyze a sentence and automatically build a **syntax tree**?

Along the way, we will cover the fundamentals of English syntax, and see that there are systematic aspects of meaning that are much easier to capture once we have identified the structure of sentences.

> NOTE: My GitHub project [automata-and-formal-language](https://github.com/dengking/automata-and-formal-language) summary the knowledge related to formal language. 

# 1  Some Grammatical Dilemmas

## 1.1  Linguistic Data and Unlimited Possibilities

In this chapter, we will adopt the formal framework of "generative grammar", in which a "language" is considered to be nothing more than an enormous collection of all grammatical sentences, and a grammar is a formal notation that can be used for "generating" the members of this set. Grammars use recursive **productions** of the form `S` → `S` *and* `S`, as we will explore in [3](http://www.nltk.org/book/ch08.html#sec-context-free-grammar). In [10.](http://www.nltk.org/book/ch10.html#chap-semantics) we will extend this, to automatically build up the meaning of a sentence out of the meanings of its parts.

## 1.2  Ubiquitous Ambiguity

A well-known example of ambiguity is shown in [(2)](http://www.nltk.org/book/ch08.html#ex-marx-elephant), from the Groucho Marx movie, *Animal Crackers* (1930):

> (2)		While hunting in Africa, I shot an elephant in my pajamas. How he got into my pajamas, I don't know.

Let's take a closer look at the ambiguity in the phrase: *I shot an elephant in my pajamas*. First we need to define a simple grammar:

```python
groucho_grammar = nltk.CFG.fromstring("""S -> NP VP
PP -> P NP
NP -> Det N | Det N PP | 'I'
VP -> V NP | VP PP
Det -> 'an' | 'my'
N -> 'elephant' | 'pajamas'
V -> 'shot'
P -> 'in'
""")
```

> NOTE: What dose NP, VP mean? see 
>
> - [2  What's the Use of Syntax?](2  What's the Use of Syntax?)
> - [3  Context Free Grammar](./3  Context Free Grammar)

This grammar permits the sentence to be analyzed in two ways, depending on whether the prepositional phrase *in my pajamas* describes the elephant or the shooting event.

```python
sent = ['I', 'shot', 'an', 'elephant', 'in', 'my', 'pajamas']
parser = nltk.ChartParser(groucho_grammar)
for tree in parser.parse(sent):
	print(tree)
```

The program produces two bracketed structures, which we can depict as trees, as shown in [(3b)](http://www.nltk.org/book/ch08.html#ex-elephant):

(3)

| a.   | ![tree_images/ch08-tree-1.png](http://www.nltk.org/book/tree_images/ch08-tree-1.png) |
| ---- | ------------------------------------------------------------ |
| b.   | ![tree_images/ch08-tree-2.png](http://www.nltk.org/book/tree_images/ch08-tree-2.png) |



Notice that there's no ambiguity concerning the meaning of any of the words;

This chapter presents grammars and parsing, as the formal and computational methods for investigating and modeling the linguistic phenomena we have been discussing. As we shall see, patterns of well-formedness and ill-formedness in a sequence of words can be understood with respect to the phrase structure and dependencies. We can develop formal models of these structures using grammars and parsers. As before, a key motivation is natural language *understanding*. How much more of the meaning of a text can we access when we can reliably recognize the linguistic structures it contains? Having read in a text, can a program "understand" it enough to be able to answer simple questions about "what happened" or "who did what to whom"? Also as before, we will develop simple programs to process annotated corpora and perform useful tasks.

# 2  What's the Use of Syntax?





In [2.2](http://www.nltk.org/book/ch08.html#fig-ic-diagram-labeled), we have added grammatical category labels to the words we saw in the earlier figure. The labels `NP`, `VP`, and `PP` stand for **noun phrase**, **verb phrase** and **prepositional phrase** respectively.



![../images/ic_diagram_labeled.png](http://www.nltk.org/images/ic_diagram_labeled.png)

**Figure 2.2**: Substitution of Word Sequences Plus Grammatical Categories: This diagram reproduces [2.1](http://www.nltk.org/book/ch08.html#fig-ic-diagram) along with grammatical categories corresponding to noun phrases (`NP`), verb phrases (`VP`), prepositional phrases (`PP`), and nominals (`Nom`).

If we now strip out the words apart from the topmost row, add an `S` node, and flip the figure over, we end up with a standard phrase structure tree, shown in [(8)](http://www.nltk.org/book/ch08.html#ex-phrase-structure-tree). Each node in this tree (including the words) is called a **constituent**. The **immediate constituents** of `S` are `NP` and `VP`.

(8)![tree_images/ch08-tree-3.png](http://www.nltk.org/book/tree_images/ch08-tree-3.png)

As we will see in the next section, a grammar specifies how the sentence can be subdivided into its immediate constituents, and how these can be further subdivided until we reach the level of individual words.

> Note
>
> As we saw in [1](http://www.nltk.org/book/ch08.html#sec-dilemmas), sentences can have arbitrary length. Consequently, phrase structure trees can have arbitrary *depth*. The cascaded chunk parsers we saw in [4](http://www.nltk.org/book/ch07.html#sec-recursion-in-linguistic-structure) can only produce structures of bounded depth, so chunking methods aren't applicable here.



# 3  Context Free Grammar

## 3.1  A Simple Grammar

Let's start off by looking at a simple context-free grammar. By convention, the left-hand-side of the first production is the **start-symbol** of the grammar, typically `S`, and all well-formed trees must have this symbol as their root label. In NLTK, context-free grammars are defined in the `nltk.grammar` module. In [3.1](http://www.nltk.org/book/ch08.html#code-cfg1) we define a grammar and show how to parse a simple sentence admitted by the grammar.

```python
	
grammar1 = nltk.CFG.fromstring("""
  S -> NP VP
  VP -> V NP | V NP PP
  PP -> P NP
  V -> "saw" | "ate" | "walked"
  NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
  Det -> "a" | "an" | "the" | "my"
  N -> "man" | "dog" | "cat" | "telescope" | "park"
  P -> "in" | "on" | "by" | "with"
  """)
```

```python
sent = "Mary saw Bob".split()
rd_parser = nltk.RecursiveDescentParser(grammar1)
for tree in rd_parser.parse(sent):
	print(tree)
	
```

The grammar in [3.1](http://www.nltk.org/book/ch08.html#code-cfg1) contains productions involving various syntactic categories, as laid out in [3.1](http://www.nltk.org/book/ch08.html#tab-syncat).

**Table 3.1**:

Syntactic Categories

| Symbol | Meaning              | Example            |
| ------ | -------------------- | ------------------ |
| S      | sentence             | *the man walked*   |
| NP     | noun phrase          | *a dog*            |
| VP     | verb phrase          | *saw a park*       |
| PP     | prepositional phrase | *with a telescope* |
| Det    | determiner           | *the*              |
| N      | noun                 | *dog*              |
| V      | verb                 | *walked*           |
| P      | preposition          | *in*               |

A production like `VP -> V NP | V NP PP` has a disjunction on the righthand side, shown by the `|` and is an abbreviation for the two productions `VP -> V NP` and `VP -> V NP PP`.

![../images/parse_rdparsewindow.png](http://www.nltk.org/images/parse_rdparsewindow.png)

**Figure 3.2**: Recursive Descent Parser Demo: This tool allows you to watch the operation of a recursive descent parser as it grows the parse tree and matches it against the input words.

> Note
>
> **Your Turn:** Try developing a simple grammar of your own, using the recursive descent parser application, `nltk.app.rdparser()`, shown in [3.2](http://www.nltk.org/book/ch08.html#fig-parse-rdparsewindow). It comes already loaded with a sample grammar, but you can edit this as you please (using the `Edit` menu). Change the grammar, and the sentence to be parsed, and run the parser using the *autostep* button.

If we parse the sentence The dog saw a man in the park using the grammar shown in [3.1](http://www.nltk.org/book/ch08.html#code-cfg1), we end up with two trees, similar to those we saw for [(3b)](http://www.nltk.org/book/ch08.html#ex-elephant):

| a.   |      | ![tree_images/ch08-tree-4.png](http://www.nltk.org/book/tree_images/ch08-tree-4.png) |
| ---- | ---- | ------------------------------------------------------------ |
| b.   |      | ![tree_images/ch08-tree-5.png](http://www.nltk.org/book/tree_images/ch08-tree-5.png) |



## 3.2  Writing Your Own Grammars

If you are interested in experimenting with writing CFGs, you will find it helpful to create and edit your grammar in a text file, say `mygrammar.cfg`. You can then load it into NLTK and parse with it as follows:

```python
grammar1 = nltk.data.load('file:mygrammar.cfg')
sent = "Mary saw Bob".split()
rd_parser = nltk.RecursiveDescentParser(grammar1)
for tree in rd_parser.parse(sent):
     print(tree)
```



## 3.3  Recursion in Syntactic Structure

A grammar is said to be **recursive** if a category occurring on the left hand side of a production also appears on the righthand side of a production, as illustrated in [3.3](http://www.nltk.org/book/ch08.html#code-cfg2). The production `Nom -> Adj Nom` (where `Nom` is the category of nominals) involves **direct recursion** on the category `Nom`, whereas **indirect recursion** on `S` arises from the combination of two productions, namely `S -> NP VP` and `VP -> V S`.

```python
	
grammar2 = nltk.CFG.fromstring("""
  S  -> NP VP
  NP -> Det Nom | PropN
  Nom -> Adj Nom | N
  VP -> V Adj | V NP | V S | V NP PP
  PP -> P NP
  PropN -> 'Buster' | 'Chatterer' | 'Joe'
  Det -> 'the' | 'a'
  N -> 'bear' | 'squirrel' | 'tree' | 'fish' | 'log'
  Adj  -> 'angry' | 'frightened' |  'little' | 'tall'
  V ->  'chased'  | 'saw' | 'said' | 'thought' | 'was' | 'put'
  P -> 'on'
  """)
```

[**Example 3.3 (code_cfg2.py)**](http://www.nltk.org/book/pylisting/code_cfg2.py): **Figure 3.3**: A Recursive Context-Free Grammar

To see how recursion arises from this grammar, consider the following trees. [(10a)](http://www.nltk.org/book/ch08.html#ex-recnominals) involves nested nominal phrases, while [(10b)](http://www.nltk.org/book/ch08.html#ex-recsentences) contains nested sentences.

(10)

| a.   |      | ![tree_images/ch08-tree-6.png](http://www.nltk.org/book/tree_images/ch08-tree-6.png) |
| ---- | ---- | ------------------------------------------------------------ |
| b.   |      | ![tree_images/ch08-tree-7.png](http://www.nltk.org/book/tree_images/ch08-tree-7.png) |

We've only illustrated two levels of recursion here, but there's no upper limit on the depth. You can experiment with parsing sentences that involve more deeply nested structures. Beware that the `RecursiveDescentParser` is unable to handle **left-recursive** productions of the form `X -> X Y`; we will return to this in [4](http://www.nltk.org/book/ch08.html#sec-parsing).

# 4  Parsing With Context Free Grammar

A **parser** processes input sentences according to the productions of a grammar, and builds one or more constituent structures that conform to the grammar. A grammar is a declarative specification of well-formedness — it is actually just a string, not a program. A parser is a procedural interpretation of the grammar. It searches through the space of trees licensed by a grammar to find one that has the required sentence along its fringe.

A parser permits a grammar to be evaluated against a collection of test sentences, helping linguists to discover mistakes in their grammatical analysis. A parser can serve as a model of psycholinguistic processing, helping to explain the difficulties that humans have with processing certain syntactic constructions. Many natural language applications involve parsing at some point; for example, we would expect the natural language questions submitted to a question-answering system to undergo parsing as an initial step.In this section we see two simple parsing algorithms, a top-down method called recursive descent parsing, and a bottom-up method called shift-reduce parsing. We also see some more sophisticated algorithms, a top-down method with bottom-up filtering called **left-corner parsing**, and a **dynamic programming** technique called **chart parsing**.

## 4.1  Recursive Descent Parsing



## 4.2  Shift-Reduce Parsing



## 4.3  The Left-Corner Parser



## 4.4  Well-Formed Substring Tables





# 5  Dependencies and Dependency Grammar

# 6  Grammar Development

Parsing builds trees over sentences, according to a phrase structure grammar. Now, all the examples we gave above only involved toy grammars containing a handful of productions. What happens if we try to scale up this approach to deal with realistic corpora of language? In this section we will see how to access treebanks, and look at the challenge of developing broad-coverage grammars.

## 6.1  Treebanks and Grammars



## 6.2  Pernicious Ambiguity



## 6.3  Weighted Grammar



# 7  Summary

