# JAX: The Missing Manual (for Research)

This guide serves as a practical, "cheatsheet-style" reference for using JAX. It focuses on the mental models required to use it effectively, specifically addressing common confusion points like dimensions, `vmap`, and state management.

## 1. The Core Mental Model

JAX is not just "NumPy with GPU support." It is a **compiler for function transformations**.

- **Pure Functions:** JAX assumes your functions are _pure_: same input, same output; no side effects (no printing\*, no global state mutation).
- **Static vs. Traced:** JAX traces your code once with "abstract tracers" (placeholders) to build a computation graph, then compiles it (XLA). Python control flow ( `if`, `for`) happens _during tracing_. If it depends on data, it vanishes unless you use JAX primitives (`jax.lax.cond`, `jax.lax.scan`).

---

## 2. Managing State (The `State` Pattern)

`equinox.Module` is a **PyTree**.

### What is a PyTree?

A PyTree is just a container of arrays (lists, tuples, dicts, custom classes). JAX functions like `jit`, `grad`, and `vmap` operate on the _leaves_ (the arrays inside) while preserving the _structure_.

### Your `State` Object

```python
import jax.numpy as jnp
import equinox as eqx

class State(eqx.Module):
    # These are the leaves (arrays)
    pos: jnp.ndarray
    vel: jnp.ndarray

    # This is static metadata (ignored by JAX derivatives/batching)
    step_count: int = eqx.field(static=True)

# Initialization
state = State(
    pos=jnp.zeros(2),
    vel=jnp.zeros(2),
    step_count=0
)
```

### The "Functional Update" Rule

In JAX, **you never mutate objects in place.** You always return a new object.

**❌ BAD (Mutating - Won't work in JIT):**

```python
def update(state):
    state.pos += 0.1  # Error! JAX arrays are immutable.
    return state
```

**✅ GOOD (Functional - Returns new PyTree):**

```python
import equinox as eqx

@jax.jit
def update(state):
    # 'eqx.tree_at' is a helper to update PyTrees functionally
    # It says: "Return a copy of 'state' where 'pos' is replaced by 'new_pos'"
    new_pos = state.pos + 0.1
    state = eqx.tree_at(lambda s: s.pos, state, new_pos)
    return state
```

---

## 3. `vmap` and `in_axes`

`jax.vmap` (Vectorizing Map) transforms a function meant for a _single_ sample into a function that handles a _batch_ of samples.

**The Golden Rule:** `vmap` pushes a dimension down. If your input has shape `(10, 3)` and you `vmap` over it, inside the function, JAX sees shape `(3)`.

### Understanding `in_axes`

`in_axes` tells JAX **which dimension is the batch dimension** for each argument.

Imagine a function:

```python
def step(state: State, dt: float):
    # state.pos shape: (2,)
    return state.pos + state.vel * dt
```

We want to run this on a **batch of 100 states**.

#### Scenario A: Batch of States, Shared `dt`

We have 100 states, but `dt` is the same for all of them.

```python
# Batch dimension is 0 for 'state', but 'dt' is NOT batched (None).
batched_step = jax.vmap(step, in_axes=(0, None))

# Input shapes:
# batch_states: State(pos=(100, 2), vel=(100, 2))
# dt: scalar

result = batched_step(batch_states, 0.1)
# Result shape: (100, 2)
```

#### Scenario B: Batch of States, Batch of `dt`s

Each state has its own unique time delta.

```python
# Batch dimension is 0 for 'state', AND 0 for 'dt'.
batched_step = jax.vmap(step, in_axes=(0, 0))

# Input shapes:
# batch_states: State(pos=(100, 2), vel=(100, 2))
# dts: (100,)

result = batched_step(batch_states, dts)
```

#### Scenario C: Complex PyTree Structures

If your `State` is a complex nested tree, `in_axes=0` automatically means "the 0-th dimension of _every array leaf_ inside this object".

If you have a tuple `(x, y)` and you want `x` batched but `y` shared:

```python
# Function: f(pair) -> where pair is (x, y)
# in_axes must match the structure of the arguments.
# in_axes = (0, None)  <-- WRONG, acts on the tuple itself.
# in_axes = ((0, None),) <-- CORRECT, acts on the contents of the first arg.
```

_Note: Usually, simpler signatures like `f(x, y)` are easier to manage than `f((x, y))`._

### Visualizing `in_axes` vs `out_axes`

| Parameter  | Value  | Meaning                                                                |
| :--------- | :----- | :--------------------------------------------------------------------- |
| `in_axes`  | `0`    | "Slice this input along the 0-th dimension. Pass one slice at a time." |
| `in_axes`  | `1`    | "Slice along the 1st dimension (e.g., columns)."                       |
| `in_axes`  | `None` | "Don't slice. Pass this entire object to every instance."              |
| `out_axes` | `0`    | "Stack the results along the 0-th dimension."                          |

---

## 4. Randomness (The "Key" Pattern)

JAX PRNG (Pseudo-Random Number Generation) is explicit. There is no global seed.

**The Pattern:** Always Split, Then Use.

```python
import jax.random as jr

key = jr.PRNGKey(42)

def step(state, key):
    # 1. SPLIT the key into as many as you need, plus one for the next step.
    step_key, noise_key, sub_key = jr.split(key, 3)

    # 2. USE the sub-keys
    noise = jr.normal(noise_key, shape=state.pos.shape)

    # 3. RETURN the 'step_key' for the next iteration
    return state, step_key
```

**Common Pitfall:** Reusing the same key results in identical "random" numbers.

---

## 5. Control Flow: `scan` and `cond`

### Loops: `scan` vs Python Loops

If you write a Python `for` loop, JAX unrolls it during compilation.

- **Short loop (10 steps):** Fine. Fast.
- **Long loop (10,000 steps):** Compile time explosion. Huge binary.

Use `jax.lax.scan` for long loops. It compiles into a single "loop" instruction in XLA.

**Template for `scan` (The "Carry" Pattern):**

```python
def one_step(carry, x):
    # 'carry' is your State
    # 'x' is input for this specific time step (optional)
    new_state = update(carry)
    output = compute_metric(new_state)
    return new_state, output

# Run for 1000 steps
final_state, history = jax.lax.scan(one_step, init_state, None, length=1000)
```

### Conditionals: `cond` vs Python `if`

- **Python `if`:** Evaluated at _trace time_. The condition must be known before JAX sees any data (e.g., a hyperparameter flag).
- **`jax.lax.cond`:** Evaluated at _runtime_ (on the GPU/TPU). Both branches are traced.

**Template for `cond`:**

```python
# jax.lax.cond(pred, true_fun, false_fun, *operands)

def do_reset(state):
    return State.reset()

def keep_going(state):
    return state

# If 'done' is true (at runtime), call do_reset(state), else keep_going(state)
next_state = jax.lax.cond(done, do_reset, keep_going, state)
```

_Note: Both branches must return PyTrees with the exact same structure and dtype._

---

## 6. JIT Compilation (`@jax.jit`)

JIT compiles your Python function into optimized machine code (XLA).

- **Static Arguments:** If a function argument changes the _shape_ of arrays or the _structure_ of the graph (e.g., a boolean flag for "is_training"), it must be marked static.

```python
from functools import partial

@partial(jax.jit, static_argnames=("is_training",))
def forward(params, x, is_training: bool):
    if is_training:  # This Python 'if' runs once during tracing
        return dropout(x)
    return x
```

- **Don't JIT everything:** Usually, you only JIT the top-level update step. JIT-ing tiny functions inside a loop is useless (and often harmful if the loop itself isn't JIT-ed).

---

## 7. Cheatsheet: Common Dimension Operations

Assume `x.shape = (B, H, W, C)` (Batch, Height, Width, Channel).

| Operation     | Code                               | Result Shape      | Note                  |
| :------------ | :--------------------------------- | :---------------- | :-------------------- |
| **Add Dim**   | `x[None, ...]`                     | `(1, B, H, W, C)` | Prepend dim           |
| **Add Dim**   | `x[..., None]`                     | `(B, H, W, C, 1)` | Append dim            |
| **Flatten**   | `x.reshape(B, -1)`                 | `(B, H*W*C)`      | Flatten all but batch |
| **Transpose** | `jnp.transpose(x, (0, 3, 1, 2))`   | `(B, C, H, W)`    | NHWC -> NCHW          |
| **Einsum**    | `jnp.einsum('bij,bjk->bik', A, B)` | `(B, I, K)`       | Batch MatMul          |

---

## 8. Debugging JAX

Since you can't just `print()` inside a JIT-ed function (it only prints once during tracing), use:

1.  **`jax.debug.print`**:
    ```python
    jax.debug.print("x shape: {x}", x=x.shape)
    ```
2.  **`jax.disable_jit()`**:
    Temporarily turns off JIT so you can use standard Python debuggers (`pdb`).
    ```python
    with jax.disable_jit():
        my_function(x)
    ```

---

## 9. Useful APIs / Cheatsheet

### Creation & Initialization

| Function                | Description                      | Example                        |
| :---------------------- | :------------------------------- | :----------------------------- |
| `jnp.zeros`, `jnp.ones` | Create arrays of 0s or 1s        | `jnp.zeros((2, 3))`            |
| `jnp.full`              | Create array with constant value | `jnp.full((2,), 3.14)`         |
| `jnp.eye`               | Identity matrix                  | `jnp.eye(3)`                   |
| `jnp.arange`            | Range of numbers                 | `jnp.arange(0, 10, 2)`         |
| `jnp.linspace`          | Evenly spaced numbers            | `jnp.linspace(0, 1, 11)`       |
| `jr.normal`             | Random normal (Gaussian)         | `jr.normal(key, (2, 2))`       |
| `jr.uniform`            | Random uniform [0, 1)            | `jr.uniform(key, (5,))`        |
| `jr.randint`            | Random integers                  | `jr.randint(key, (5,), 0, 10)` |

### Array Manipulation

| Function          | Description                | Example                           |
| :---------------- | :------------------------- | :-------------------------------- |
| `jnp.stack`       | Join arrays along new axis | `jnp.stack([x, y])`               |
| `jnp.concatenate` | Join along existing axis   | `jnp.concatenate([x, y], axis=0)` |
| `jnp.where`       | Conditional selection      | `jnp.where(x > 0, x, 0)`          |
| `jnp.squeeze`     | Remove dims of size 1      | `x.squeeze()`                     |
| `jnp.expand_dims` | Add dim of size 1          | `jnp.expand_dims(x, axis=0)`      |
| `jnp.ravel`       | Flatten to 1D              | `x.ravel()`                       |

### Math & Logic

| Function                | Description               | Example                  |
| :---------------------- | :------------------------ | :----------------------- |
| `jnp.dot` / `@`         | Matrix multiplication     | `A @ B`                  |
| `jnp.sum`, `jnp.mean`   | Sum/Mean (supports axis)  | `x.mean(axis=-1)`        |
| `jnp.max`, `jnp.argmax` | Max value / Index of max  | `x.argmax()`             |
| `jnp.clip`              | Clip values to range      | `jnp.clip(x, 0, 1)`      |
| `jnp.allclose`          | Check if arrays are close | `jnp.allclose(x, y)`     |
| `jax.nn.softmax`        | Softmax function          | `jax.nn.softmax(logits)` |
| `jax.nn.relu`           | ReLU activation           | `jax.nn.relu(x)`         |

### Functional Transformations

| Function             | Description                               |
| :------------------- | :---------------------------------------- |
| `jax.jit`            | Just-In-Time compilation (speed).         |
| `jax.grad`           | Computes gradient (single scalar output). |
| `jax.value_and_grad` | Returns `(value, gradient)`.              |
| `jax.vmap`           | Vectorizes a function (adds batch dim).   |
| `jax.lax.scan`       | Efficient loop with carry.                |
| `jax.lax.cond`       | Efficient conditional branching.          |

---

## 10. Einops (Tensor Magic)

Einops is not built-in, but it is the _idiomatic_ way to handle tensor reshaping in JAX research code. It is clearer and safer than `reshape`/`transpose`.

**APIs:** `rearrange`, `reduce`, `repeat`

```python
from einops import rearrange, reduce, repeat

# Assume images: (Batch, Height, Width, Channel)
x = jnp.zeros((32, 64, 64, 3))
```

### 1. Rearrange (Permute, Flatten, Split)

Replaces `transpose`, `reshape`, and `squeeze`.

```python
# HWC to CHW (PyTorch style)
x_chw = rearrange(x, 'b h w c -> b c h w')

# Flatten images to vectors
x_flat = rearrange(x, 'b h w c -> b (h w c)')
# Shape: (32, 12288)

# Split heads (for Attention)
# Assume x is (Batch, SeqLen, Dim) -> (Batch, SeqLen, Heads, HeadDim)
q = rearrange(q, 'b s (h d) -> b s h d', h=8)
```

### 2. Reduce (Pool, Mean, Max)

Replaces `mean`, `sum`, `max` with explicit axes.

```python
# Global Average Pooling (spatial mean)
# "Reduce over h and w, keep b and c"
y = reduce(x, 'b h w c -> b c', 'mean')
# Shape: (32, 3)

# Max over channels
y = reduce(x, 'b h w c -> b h w', 'max')
```

### 3. Repeat (Broadcast)

Replaces `tile` or `expand_dims`.

```python
# Repeat a class token (1, D) for every item in batch
cls_token = jnp.zeros((1, 512))
cls_tokens = repeat(cls_token, '1 d -> b d', b=32)
# Shape: (32, 512)
```

---

## 11. PyTrees & JAX Tree Functions

A **PyTree** is a container of leaf elements and/or more PyTrees. Containers include lists, tuples, dicts, `None`, and custom classes registered with JAX. Leaves are typically JAX arrays, but can be anything JAX treats as an opaque value.

### Key Concepts

- **Leaf:** An element in a PyTree that is not a PyTree itself (usually a JAX array or scalar).
- **Structure (Treedef):** The shape/container-type of the PyTree, disregarding the actual leaf values.

### Common `jax.tree` Functions with Examples

JAX provides powerful utilities for working with PyTrees under `jax.tree` (formerly `jax.tree_util`).

#### 1. `jax.tree.map` (Mapping over leaves)

Applies a function to every leaf in the PyTree, preserving the structure.

```python
import jax
import jax.numpy as jnp

tree = {'a': jnp.array([1, 2]), 'b': (jnp.array([3]), jnp.array([4, 5]))}

# Double every element in the tree
doubled_tree = jax.tree.map(lambda x: x * 2, tree)
# Result: {'a': Array([2, 4]), 'b': (Array([6]), Array([8, 10]))}

# Add two trees of the exact same structure
tree2 = {'a': jnp.array([10, 20]), 'b': (jnp.array([30]), jnp.array([40, 50]))}
sum_tree = jax.tree.map(lambda x, y: x + y, tree, tree2)
```

#### 2. `jax.tree.leaves` (Extracting leaves)

Extracts all leaves from a PyTree into a flat list. Useful for operations like computing a global metric across all parameters.

```python
tree = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0])}
leaves = jax.tree.leaves(tree)
# leaves: [Array([1., 2.]), Array([3.])]

# Compute the L2 norm of all parameters in a model
global_norm = jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree.leaves(tree)))
```

#### 3. `jax.tree.reduce` (Reducing leaves)

Reduces all leaves to a single value.

```python
tree = {'a': jnp.array([1, 2]), 'b': jnp.array([3, 4])}

# Sum of all elements across all arrays
total_sum = jax.tree.reduce(lambda acc, x: acc + jnp.sum(x), tree, 0.0)
```

#### 4. `jax.tree.flatten` and `jax.tree.unflatten`

Useful when you need to interface with APIs that only accept flat lists of arrays (e.g., `scipy.optimize`).

```python
tree = {'weights': jnp.ones((2, 2)), 'bias': jnp.zeros(2)}

# Flatten
flat_leaves, treedef = jax.tree.flatten(tree)

# ... do some flat operations ...
new_flat_leaves = [leaf * 2 for leaf in flat_leaves]

# Unflatten back to the original structure
new_tree = jax.tree.unflatten(treedef, new_flat_leaves)
```

### Custom PyTrees

If you don't use `equinox.Module` or `flax.struct.dataclass`, you must register your custom classes with JAX so it knows how to flatten and unflatten them. Otherwise, JAX treats the object itself as a single opaque leaf.

```python
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class MyTree:
    def __init__(self, a, b):
        self.a = a  # Leaf
        self.b = b  # Leaf

    def tree_flatten(self):
        # Return a tuple of (children, auxiliary_data)
        children = (self.a, self.b)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Reconstruct the object
        return cls(*children)

# Now jax.tree.map will go inside MyTree to modify 'a' and 'b'
obj = MyTree(jnp.array([1]), jnp.array([2]))
new_obj = jax.tree.map(lambda x: x * 10, obj)
```

---

## 12. Equinox Tree Operations

Equinox provides additional PyTree utilities that are essential for deep learning, specifically for managing state and model parameters.

### 1. `eqx.tree_at` (Functional Update)

The most common way to "mutate" an immutable PyTree. JAX arrays and Equinox modules are immutable; `tree_at` returns a new copy with the specified leaf modified.

```python
import equinox as eqx

# Replace a leaf
# Syntax: eqx.tree_at(where_fn, pytree, replace_with)
new_state = eqx.tree_at(lambda s: s.weights, state, new_weights)

# Apply a function to a leaf
new_state = eqx.tree_at(lambda s: s.step, state, lambda x: x + 1)
```

### 2. `eqx.partition` and `eqx.combine`

Used to separate a PyTree into parts based on a filter function. This is critical for functions like `jax.jit` or `jax.grad` which require all inputs to be JAX arrays (unless marked as static).

```python
# Partition into JAX-compatible arrays and "static" metadata (strings, bools, etc.)
arrays, static = eqx.partition(model, eqx.is_array)

# Now 'arrays' can be safely passed into JAX transformations
# 'static' is usually handled by closures or marked as static in JIT

# Reconstruct the original model structure
model = eqx.combine(arrays, static)
```

### 3. `eqx.tree_serialise_leaves` and `eqx.tree_deserialise_leaves`

The standard way to save and load Equinox models. It uses `safetensors` under the hood if available, or `numpy`.

```python
# Save model to disk
eqx.tree_serialise_leaves("model.eqx", model)

# Load from disk
# Requires a 'template' model that has the exact same structure as the saved one.
model = eqx.tree_deserialise_leaves("model.eqx", model)
```

### 4. Filtering Utilities

Equinox provides predicates to help with partitioning and mapping.

| Predicate              | Description                                                      |
| :--------------------- | :--------------------------------------------------------------- |
| `eqx.is_array`         | Matches any JAX array or scalar.                                 |
| `eqx.is_inexact_array` | Matches floating point JAX arrays (useful for trainable params). |
| `eqx.is_array_like`    | Matches arrays or things that can be converted to arrays.        |

---
