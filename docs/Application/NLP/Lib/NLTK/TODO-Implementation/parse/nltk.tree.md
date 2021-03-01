# [nltk.tree.Tree](https://github.com/nltk/nltk/blob/develop/nltk/tree.py)

相关文档有：

- [Unit tests for nltk.tree.Tree](http://www.nltk.org/howto/tree.html)

看了它的实现后，发现它的实现方式和普通的实现树的方式不同：

普通的实现方式都是定义`class Node`，然后`Node`中包含指向子节点的指针，这种方式是将树的构成看做是`Node`之间的关联；

而它显示地定义`class Node`，直接定义`class Tree`，它将树的构成看做是`Tree`之间的关联。

```python
class Tree(list):
    def __init__(self, node, children=None):
        if children is None:
            raise TypeError(
                "%s: Expected a node value and child list " % type(self).__name__
            )
        elif isinstance(children, string_types):
            raise TypeError(
                "%s() argument 2 should be a list, not a "
                "string" % type(self).__name__
            )
        else:
            list.__init__(self, children)
            self._label = node
```



显然，虽然`class Tree`叫做`Tree`，但是实际上，它的功能非常类似于`Node`。

可以看到，`class Tree`继承了`list`，它将和子树之间的关联全部保存与`list`中。

另外它只有一个成员变量`_label`，这是因为`class Tree`主要用作parse tree，parse tree中的每个node只有一个label。

下面是代码中给出的该类的文档：

> A Tree represents a hierarchical grouping of leaves and subtrees. For example, each constituent in a syntax tree is represented by a single Tree.
>
> A tree's children are encoded as a list of leaves and subtrees, where a leaf is a basic (non-tree) value; and a subtree is a nested Tree.

从上述文档中，可以看出`class Tree`和普通的树的实现方式之间的一个差异：

在普通的实现中，树中节点的类型都是相同的，这里把它称为`class Node`，无论是内节点还是叶子节点；

但是在`class Tree`中，没有按照这种方式实现，而是所有的内节点的内省都是`class Tree`，而叶子节点的类型是basic (non-tree)  type。



