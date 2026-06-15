# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import itertools as it

from jax._src import tree
from jax._src import util
from jax._src import tree_util

hole = util.Singleton("_")

@staticmethod
def flatten_static_argnums_argnames(
    args, kwargs, static_argnums, static_argnames):
  assert not static_argnums
  assert not static_argnames
  return pack(*(flatten(arg) for arg in args))

def flat_list(xs): return FTList(xs)
def flatten(pytree):
  xs, treedef = tree.flatten(pytree)
  return FTPyTree(xs, treedef)
def pack(*trees): return FTTuple(trees)

class FlatTree:
  """FlatTree is a Python OOP version of this functor:

    data FlatTree a = Tuple [FlatTree a]
                    | WithAux (FlatTree a) Aux
                    | List [a]
                    | Filtered [a] (FlatTree (Either Hole Aux))
                    | Pytree [a] PyTreeDef
                    | ArgsAndKwargs [Either Aux (FlatTree a)]
                                    {Either Aux (FlatTree a)]
  """
  def unpack(self): raise TypeError(f"Not a FlatTree tuple: {self}")
  def from_list(self): raise TypeError(f"Not a FlatTree list: {self}")
  def unflatten(self): raise TypeError(f"Not a flattened PyTree: {self}")
  def unfilter(self): raise TypeError(f"Not a filtered FlatTree: {self}")
  def map(self, f): return self.update(map(f, self))
  def map2(self, ys, f): return self.update(map(f, self, ys))
  def map3(self, ys, zs, f): return self.update(map(f, self, ys, zs))
  def with_aux(self, aux): return FTWithAux(self, aux)
  def update(self, xs):
    xs = list(xs)
    assert len(xs) == len(self)
    xs_iter = iter(xs)
    return self._iter_update(xs_iter)

  def filter(self, keeps): return self.map2(keeps, lambda x, k: (x, k))._filter()
  def _filter(self):
    kept = [x for x, keep in self if keep]
    put_aside = self.map(lambda x_keep: hole if x_keep[1] else x_keep[0])
    return FTFiltered(kept, put_aside)

class FTTuple(FlatTree):

  def __init__(self, trees):
    trees = trees if isinstance(trees, tuple) else tuple(trees)
    for t in trees: assert isinstance(t, FlatTree)
    self.trees = trees
  def unpack(self): return self.trees
  def _filter(self): return FTTuple(t._filter() for t in self.trees)
  def unfilter(self): return FTTuple(t.unfilter() for t in self.trees)
  def __iter__(self): return (x for tree in self.trees for x in tree)
  def __len__(self): return sum(len(t) for t in self.trees)
  def _iter_update(self, xs_iter):
    return FTTuple(tuple(t._iter_update(xs_iter) for t in self.trees))
  def __repr__(self): return repr(self.trees)
  def __eq__(self, other): return isinstance(other, FTTuple) and self.trees == other.trees
  def __hash__(self): return hash(self.trees)

class FTWithAux(FlatTree):
  def __init__(self, ft, aux):
    assert isinstance(ft, FlatTree)
    self.ft = ft
    self.aux = aux
  def __iter__(self): return iter(self.ft)
  def __len__(self): return len(self.ft)
  def _iter_update(self, xs_iter):
    return FTWithAux(self.ft._iter_update(xs_iter), self.aux)
  def __repr__(self): return f"WithAux(ft={self.ft}, aux={self.aux})"
  def __eq__(self, other):
    return (isinstance(other, FTWithAux) and
            self.ft == other.ft and self.aux == other.aux)
  def __hash__(self): return hash((self.xs, self.aux))

class FTList(FlatTree):
  def __init__(self, xs):
    xs = xs if isinstance(xs, tuple) else tuple(xs)
    self.xs = xs
  def from_list(self): return list(self.xs)
  def __iter__(self): return iter(self.xs)
  def __len__(self): return len(self.xs)
  def _iter_update(self, xs_iter): return FTList(it.islice(xs_iter, len(self.xs)))
  def __repr__(self): return repr(list(self.xs))
  def __eq__(self, other): return isinstance(other, FTList) and self.xs == other.xs
  def __hash__(self): return hash(self.xs)

class FTFiltered(FlatTree):
  def __init__(self, xs, ft_statics):
    assert isinstance(ft_statics, FlatTree)
    self.xs = xs
    self.ft_statics = ft_statics
  def unfilter(self):
    xs_iter = iter(self.xs)
    return self.ft_statics.map(lambda s: next(xs_iter) if s is hole else s)
  def __iter__(self): return iter(self.xs)
  def __len__(self): return len(self.xs)
  def _iter_update(self, xs_iter): return FTList(it.islice(xs_iter, len(self.xs)))
  def __repr__(self): return f"Filtered(vals={self.xs}, ft={self.ft_statics})"
  def __eq__(self, other):
    return (isinstance(other, FTFiltered) and
            self.xs == other.xs and self.ft_statics == other.ft_statics)
  def __hash__(self): return hash((self.xs, self.ft_statics))

class FTPyTree(FlatTree):
  def __init__(self, xs, treedef):
    assert isinstance(treedef, tree_util.PyTreeDef)
    xs = xs if isinstance(xs, tuple) else tuple(xs)
    self.xs = xs
    self.treedef = treedef

  def unflatten(self):
    return tree_util.tree_unflatten(self.treedef, self.xs)
  def __iter__(self): return iter(self.xs)
  def __len__(self): return len(self.xs)
  def _iter_update(self, xs_iter):
    return FTPyTree(it.islice(xs_iter, len(self.xs)), self.treedef)
  def __repr__(self): return f"Pytree(vals={list(self.xs)}, tree={self.treedef})"
  def __eq__(self, other):
    return (isinstance(other, FTPyTree) and
            self.xs == other.xs and self.treedef == other.treedef)
  def __hash__(self): return hash((self.xs, self.treedef))

  @property
  def paths(self) -> FlatTree:
    # TODO(dougalm): find a way to do this without roundtripping
    try:
      paths, _ = unzip2(self.registry.flatten_with_path(self.unflatten())[0])
      assert len(paths) == len(self.xs)
      return self.update(paths)
    except:
      return self.update([()] * len(self.xs))  # not our fault
