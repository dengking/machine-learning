[TOC]

# 7. Extracting Information from Text

For any given question, it's likely that someone has written the answer down somewhere. The amount of natural language text that is available in electronic form is truly staggering, and is increasing every day. However, the complexity of natural language can make it very difficult to access the information in that text. The state of the art in NLP is still a long way from being able to build general-purpose representations of meaning from unrestricted text. If we instead focus our efforts on a limited set of questions or "entity relations," such as "where are different facilities located," or "who is employed by what company," we can make significant progress. The goal of this chapter is to answer the following questions:

1. How can we build a system that extracts structured data, such as tables, from unstructured text?
2. What are some robust methods for identifying the entities and relationships described in a text?
3. Which corpora are appropriate for this work, and how do we use them for training and evaluating our models?

Along the way, we'll apply techniques from the last two chapters to the problems of **chunking** and **named-entity recognition**.

# 1  Information Extraction

Information comes in many shapes and sizes. One important form is **structured data**, where there is a regular and predictable organization of **entities** and **relationships**. For example, we might be interested in the relation between companies and locations. Given a particular company, we would like to be able to identify the locations where it does business; conversely, given a location, we would like to discover which companies do business in that location. If our data is in tabular form, such as the example in [1.1](http://www.nltk.org/book/ch07.html#tab-db-locations), then answering these queries is straightforward.

**Table 1.1**:

Locations data

| OrgName             | LocationName |
| ------------------- | ------------ |
| Omnicom             | New York     |
| DDB Needham         | New York     |
| Kaplan Thaler Group | New York     |
| BBDO South          | Atlanta      |
| Georgia-Pacific     | Atlanta      |

Things are more tricky if we try to get similar information out of text. For example, consider the following snippet (from nltk.corpus.ieer, for fileid NYT19980315.0085).

> (1)		The fourth Wells account moving to another agency is the packaged paper-products division of Georgia-Pacific Corp., which arrived at Wells only last fall. Like Hertz and the History Channel, it is also leaving for an Omnicom-owned agency, the BBDO South unit of BBDO Worldwide. BBDO South in Atlanta, which handles corporate advertising for Georgia-Pacific, will assume additional duties for brands like Angel Soft toilet tissue and Sparkle paper towels, said Ken Haldin, a spokesman for Georgia-Pacific in Atlanta.

If you read through [(1)](http://www.nltk.org/book/ch07.html#ex-ie4), you will glean the information required to answer the example question. But how do we get a machine to understand enough about [(1)](http://www.nltk.org/book/ch07.html#ex-ie4) to return the answers in [1.2](http://www.nltk.org/book/ch07.html#tab-db-answers)? This is obviously a much harder task. Unlike [1.1](http://www.nltk.org/book/ch07.html#tab-db-locations), [(1)](http://www.nltk.org/book/ch07.html#ex-ie4) contains no **structure** that links organization names with location names.

One approach to this problem involves building a very general representation of meaning ([10.](http://www.nltk.org/book/ch10.html#chap-semantics)). In this chapter we take a different approach, deciding in advance that we will only look for very specific kinds of information in text, such as the relation between organizations and locations. Rather than trying to use text like [(1)](http://www.nltk.org/book/ch07.html#ex-ie4) to answer the question directly, we first convert the **unstructured data** of natural language sentences into the **structured data** of [1.1](http://www.nltk.org/book/ch07.html#tab-db-locations). Then we reap the benefits of powerful query tools such as SQL. This method of getting meaning from text is called **Information Extraction**.

**Information Extraction** has many applications, including business intelligence, resume harvesting, media analysis, sentiment detection, patent search, and email scanning. A particularly important area of current research involves the attempt to extract structured data out of electronically-available scientific literature, especially in the domain of biology and medicine.

## 1.1  Information Extraction Architecture

[1.1](http://www.nltk.org/book/ch07.html#fig-ie-architecture) shows the architecture for a simple **information extraction system**. It begins by processing a document using several of the procedures discussed in [3](http://www.nltk.org/book/ch03.html#chap-words) and [5.](http://www.nltk.org/book/ch05.html#chap-tag): first, the raw **text** of the document is split into **sentences** using a **sentence segmenter**, and each **sentence** is further subdivided into **words** using a **tokenizer**. Next, each sentence is tagged with **part-of-speech tags**, which will prove very helpful in the next step, **named entity detection**. In this step, we search for mentions of potentially interesting entities in each sentence. Finally, we use **relation detection** to search for likely relations between different entities in the text.

![../images/ie-architecture.png](http://www.nltk.org/images/ie-architecture.png)

**Figure 1.1**: Simple Pipeline Architecture for an Information Extraction System. This system takes the raw text of a document as its input, and generates a list of `(entity, relation, entity)` tuples as its output. For example, given a document that indicates that the company Georgia-Pacific is located in Atlanta, it might generate the tuple `([ORG: 'Georgia-Pacific'] 'in' [LOC: 'Atlanta'])`.

To perform the first three tasks, we can define a simple function that simply connects together NLTK's default sentence segmenter [1], word tokenizer [2], and part-of-speech tagger [3]:

```python
 	
>>> def ie_preprocess(document):
...    sentences = nltk.sent_tokenize(document) [1]
...    sentences = [nltk.word_tokenize(sent) for sent in sentences] [2]
...    sentences = [nltk.pos_tag(sent) for sent in sentences] [3]
```

Next, in **named entity detection**, we segment and label the entities that might participate in interesting relations with one another. Typically, these will be definite noun phrases such as *the knights who say "ni"*, or proper names such as *Monty Python*. In some tasks it is useful to also consider indefinite nouns or noun chunks, such as every student or cats, and these do not necessarily refer to entities in the same way as definite `NP`s and proper names.

Finally, in relation extraction, we search for specific patterns between pairs of entities that occur near one another in the text, and use those patterns to build tuples recording the relationships between the entities.

# 2  Chunking

The basic technique we will use for entity detection is **chunking**, which segments and labels multi-token sequences as illustrated in [2.1](http://www.nltk.org/book/ch07.html#fig-chunk-segmentation). The smaller boxes show the word-level tokenization and part-of-speech tagging, while the large boxes show higher-level chunking. Each of these larger boxes is called a **chunk**. Like tokenization, which omits whitespace, chunking usually selects a subset of the tokens. Also like tokenization, the pieces produced by a chunker do not overlap in the source text.



![../images/chunk-segmentation.png](http://www.nltk.org/images/chunk-segmentation.png)**Figure 2.1**: Segmentation and Labeling at both the Token and Chunk Levels

In this section, we will explore chunking in some depth, beginning with the definition and representation of chunks. We will see regular expression and n-gram approaches to chunking, and will develop and evaluate chunkers using the CoNLL-2000 chunking corpus. We will then return in [(5)](http://www.nltk.org/book/ch07.html#sec-ner) and [6](http://www.nltk.org/book/ch07.html#sec-relextract) to the tasks of named entity recognition and relation extraction.

## 2.1  Noun Phrase Chunking

We will begin by considering the task of **noun phrase chunking**, or **NP-chunking**, where we search for chunks corresponding to individual noun phrases. For example, here is some Wall Street Journal text with `NP`-chunks marked using brackets:

> (2)		[ The/DT market/NN ] for/IN [ system-management/NN software/NN ] for/IN [ Digital/NNP ] [ 's/POS hardware/NN ] is/VBZ fragmented/JJ enough/RB that/IN [ a/DT giant/NN ] such/JJ as/IN [ Computer/NNP Associates/NNPS ] should/MD do/VB well/RB there/RB ./.

As we can see, `NP`-chunks are often smaller pieces than complete noun phrases. For example, *the market for system-management software for Digital's hardware* is a single noun phrase (containing two nested noun phrases), but it is captured in `NP`-chunks by the simpler chunk *the market*. One of the motivations for this difference is that `NP`-chunks are defined so as not to contain other `NP`-chunks. Consequently, any prepositional phrases or subordinate clauses that modify a nominal will not be included in the corresponding `NP`-chunk, since they almost certainly contain further noun phrases.

One of the most useful sources of information for `NP`-chunking is **part-of-speech tags**. This is one of the motivations for performing **part-of-speech tagging** in our information extraction system. We demonstrate this approach using an example sentence that has been part-of-speech tagged in [2.2](http://www.nltk.org/book/ch07.html#code-chunkex). In order to create an `NP`-chunker, we will first define a **chunk grammar**, consisting of rules that indicate how sentences should be chunked. In this case, we will define a simple grammar with a single regular-expression rule. 

This rule says that an NP chunk should be formed whenever the chunker finds an optional determiner (`DT`) followed by any number of adjectives (`JJ`) and then a noun (`NN`). Using this grammar, we create a chunk parser, and test it on our example sentence. The result is a tree, which we can either print, or display graphically.

```python
sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"), ("dog", "NN"), ("barked", "VBD"), ("at", "IN"),  ("the", "DT"), ("cat", "NN")]

grammar = "NP: {<DT>?<JJ>*<NN>}" 

cp = nltk.RegexpParser(grammar)
result = cp.parse(sentence) 
print(result) 
(S
  (NP the/DT little/JJ yellow/JJ dog/NN)
  barked/VBD
  at/IN
  (NP the/DT cat/NN))
result.draw() 
```

[**Example 2.2 (code_chunkex.py)**](http://www.nltk.org/book/pylisting/code_chunkex.py): **Figure 2.2**: Example of a Simple Regular Expression Based NP Chunker.

![tree_images/ch07-tree-1.png](http://www.nltk.org/book/tree_images/ch07-tree-1.png)

> NOTE: In fact, this is a parse tree.

## 2.2  Tag Patterns

The rules that make up a chunk grammar use **tag patterns** to describe sequences of tagged words. A tag pattern is a sequence of part-of-speech tags delimited using angle brackets, e.g. `?*`. Tag patterns are similar to regular expression patterns ([3.4](http://www.nltk.org/book/ch03.html#sec-regular-expressions-word-patterns)). Now, consider the following noun phrases from the Wall Street Journal:

```
another/DT sharp/JJ dive/NN
trade/NN figures/NNS
any/DT new/JJ policy/NN measures/NNS
earlier/JJR stages/NNS
Panamanian/JJ dictator/NN Manuel/NNP Noriega/NNP
```

We can match these noun phrases using a slight refinement of the first tag pattern above, i.e. `?*+`. This will chunk any sequence of tokens beginning with an optional determiner, followed by zero or more adjectives of any type (including relative adjectives like `earlier/JJR`), followed by one or more nouns of any type. However, it is easy to find many more complicated examples which this rule will not cover:

```
his/PRP$ Mansion/NNP House/NNP speech/NN
the/DT price/NN cutting/VBG
3/CD %/NN to/TO 4/CD %/NN
more/JJR than/IN 10/CD %/NN
the/DT fastest/JJS developing/VBG trends/NNS
's/POS skill/NN
```

Note

**Your Turn:** Try to come up with tag patterns to cover these cases. Test them using the graphical interface `nltk.app.chunkparser()`. Continue to refine your tag patterns with the help of the feedback given by this tool.